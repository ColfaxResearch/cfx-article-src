"""
==================================
NVFP4 Block-Scaled GEMM -- V5
==================================

Originally converted from a CUTLASS dense gemm example. 
"""



import argparse
from typing import Optional, Tuple, Type, Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import make_ptr
import functools
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField


# Bank conflict-free gmem/smem set up.

def sm120_make_smem_layout_sfa_cf(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
) -> cute.Layout:
    assert isinstance(tile_shape_mnk, tuple)

    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    # Conflict-free atom: m0 stride 4, m1 stride 32*4 (word index == row index)
    mn_basic_block_shape = (32, blk_mn // 32)
    mn_basic_block_stride = (blk_sf, 32 * blk_sf)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % blk_mn == 0, (
        "tile_shape_mnk[0] must be divisible by blk_mn"
    )

    sSFA_shapeM = (mn_basic_block_shape, tile_shape_mnk[0] // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        tile_shape_mnk[0] // blk_mn * blk_elems,
    )

    smem_layout = cute.make_layout(
        (sSFA_shapeM, sSFA_shapeK), stride=(sSF_strideM, sSF_strideK)
    )

    # (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )
    print("sfa_smem_layout_staged",sfa_smem_layout_staged)

    return sfa_smem_layout_staged


def tile_atom_to_shape_SF_cf(shape, sf_vec_size):
    chunk = cute.make_layout(
        ((32, 4), (sf_vec_size, 4)),
        stride=((4, 32 * 4), (0, 1)),
    )
    return cute.tile_to_shape(chunk, shape, (2, 1, 3))


class BlockscaledGemmKernel:

    def __init__(
        self,
        sf_vec_size: int,
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        swizzle_size: int = 32,
        epi_tile_m: int = 64,
        epi_tile_n: int = 128,
        ab_stage: Optional[int] = 2,
        epi_stage: Optional[int] = 1,
        raster_along_m: bool = True,
        sf_tma_internal_type: str = "int16",
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        tile_m, tile_n, tile_k = self.tile_shape_mnk
        self.sfa_tile_shape_mk = (max(128, tile_m), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // tile_m
        self.sfb_tile_shape_nk = (max(128, tile_n), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // tile_n
        self.cluster_shape_mnk = (1, 1, 1)  
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        sf_tma_internal_type = sf_tma_internal_type.lower()
        assert sf_tma_internal_type in ("none", "int16"), (
            "sf_tma_internal_type must be 'none' or 'int16'"
        )
        self.sf_tma_internal_type = sf_tma_internal_type
        if epi_tile_n in (0, None):
            epi_n = tile_n
        else:
            epi_n = min(epi_tile_n, tile_n)
        if epi_tile_m in (0, None):
            epi_m = tile_m
        else:
            epi_m = min(epi_tile_m, tile_m)
        assert tile_n % epi_n == 0 and epi_n % 16 == 0, (
            "epi_tile_n must divide tile N and be a multiple of 16"
        )
        assert tile_m % epi_m == 0 and epi_m % 16 == 0, (
            "epi_tile_m must divide tile M and be a multiple of 16"
        )
        self.epi_tile = (epi_m, epi_n)
        self.ab_stage_override = ab_stage
        self.epi_stage_override = epi_stage
        self.tiled_mma = None
        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.tma_store_warp_id = self.num_mma_warps + 1
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 4  
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.epilog_free_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=(self.num_mma_warps + 1) * self.num_threads_per_warp,
        )
        self.epilog_ready_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=(self.num_mma_warps + 1) * self.num_threads_per_warp,
        )
        self.load_register_requirement = 32
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((4, 2, 1)),
            permutation_mnk=permutation_mnk,
        )

        warp_shape = (4,2,1)

        # MMA atom: m16, n8, k64; atom_layout: (4,2,1) -> group: m64, n16, k64
        mma_m, mma_n, mma_k = 16, 8, 64
        self.num_m_tiles = self.tile_shape_mnk[0] // (mma_m * warp_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (mma_n * warp_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa_cf(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_per_stage = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        computed_ab_stage, self.epi_stage, computed_epi_stage_max = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.epi_stage_override,
        )

        assert self.epi_stage > 0, (
            "epi_stage <= 0, not enough shared memory. This configuration will be skipped."
        )

        if self.ab_stage_override is None:
            self.ab_stage = computed_ab_stage
        else:
            assert 1 <= self.ab_stage_override <= computed_ab_stage, (
                f"ab_stage={self.ab_stage_override} exceeds the computed legal maximum "
                f"{computed_ab_stage} for tile_shape_mnk={self.tile_shape_mnk}, "
                f"epi_tile={self.epi_tile}"
            )
            self.ab_stage = self.ab_stage_override

        print(
            f"Pipeline config: ab_stage={self.ab_stage} "
            f"(computed_max={computed_ab_stage}), "
            f"epi_tile={self.epi_tile}, epi_stage={self.epi_stage} "
            f"(epi_stage_max={computed_epi_stage_max}), "
            f"scheduler swizzle_size={self.swizzle_size}, "
            f"raster_along_m={self.raster_along_m}, "
            f"sf_tma_internal_type={self.sf_tma_internal_type}"
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A
            b: Input tensor B
            sfa: Scale factor tensor for A
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        self.sfa_layout = tile_atom_to_shape_SF_cf(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        sf_tma_internal_type = (
            cutlass.Int16 if self.sf_tma_internal_type == "int16" else None
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=sf_tma_internal_type,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=sf_tma_internal_type,
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            self.sfa_tile_shape_mk,
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # TMA partitions for SFA
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA_full = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)

        tCrSFB_full = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        tCrSFA_full = cute.group_modes(tCrSFA_full, 2, cute.rank(tCrSFA_full))
        tCrSFB_full = cute.group_modes(tCrSFB_full, 2, cute.rank(tCrSFB_full))
        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sfa_tile_offset, 0, None),
                    )
                    tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA_tile)
                    tCrSFA_tile = sm120_utils.partition_fragment_SFA(
                        sSFA_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                else:
                    tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                    tCrSFA_tile = tCrSFA_full
                    tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                    sSFB_tile = cute.local_tile(
                        sSFB,
                        cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (sfb_tile_offset, 0, None),
                    )
                    tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB_tile)
                    tCrSFB_tile = sm120_utils.partition_fragment_SFB(
                        sSFB_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                else:
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                cute.copy(
                    smem_tiled_copy_SFA,
                    tCsSFA_p_filtered[None, None, 0],
                    tCrSFA_copy_view_filtered[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_SFB,
                    tCsSFB_p_filtered[None, None, 0],
                    tCrSFB_copy_view_filtered[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=2):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA_tile[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                tCrSFB_tile[None, None, k_block_idx],
                            ],
                            accumulators,
                        )

                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(
                            tCrSFA_tile_copy_view
                        )
                        tCrSFB_copy_view_filtered = cute.filter_zeros(
                            tCrSFB_tile_copy_view
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )


                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(
                            tCrSFA_tile_copy_view
                        )
                        tCrSFB_copy_view_filtered = cute.filter_zeros(
                            tCrSFB_tile_copy_view
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )

                    cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA_tile[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                tCrSFB_tile[None, None, k_block_idx],
                            ],
                            accumulators,
                        )

                # EPILOGUE
                _is_m_major = self.c_layout.is_m_major_c()
                if cutlass.const_expr(self.c_dtype.width == 16):
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                        self.c_dtype,
                    )
                else:
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                    )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        2,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
                epi_rest_n = self.tile_shape_mnk[1] // self.epi_tile[1]
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                has_multi_epi_store = cutlass.const_expr(
                    not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
                )
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        MmaMPerEpiM = epi_tile_m // mma_tile_m
                        MmaNPerEpiN = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]

                        # Type conversion with alpha scaling
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        acc_vec = tRS_rD.load()
                        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                        tRS_rD_out.store(acc_vec)

                        # Register to shared memory
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                            tRS_sD, mode=[3]
                        )

                        if has_multi_epi_store:
                            self.epilog_free_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_ready_barrier.arrive()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # Dedicated TMA store warp
        elif warp_idx == self.tma_store_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            epi_rest_m = self.tile_shape_mnk[0] // self.epi_tile[0]
            epi_rest_n = self.tile_shape_mnk[1] // self.epi_tile[1]
            has_multi_epi_store = cutlass.const_expr(
                not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
            )

            sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_threads_per_warp,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                tcgc_for_tma_partition = cute.zipped_divide(
                    gC_mnl_slice, self.epi_tile
                )
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        epi_buffer = (
                            epi_m * epi_rest_n + epi_n
                        ) % self.epi_stage


                        if has_multi_epi_store:
                            tma_store_pipeline.producer_acquire()
                            self.epilog_free_barrier.arrive()

                        self.epilog_ready_barrier.arrive_and_wait()

                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, (epi_m, epi_n))],
                        )

                        if has_multi_epi_store:
                            tma_store_pipeline.producer_commit()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

        # DMA warp group
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                tAgSFA_mkl = tAgSFA[(None, sfa_tile_coord_m, None, tile_coord_mnl[2])]
                sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                tBgSFB_nkl = tBgSFB[(None, sfb_tile_coord_n, None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for _k_tile in range(0, k_tile_cnt, 1, unroll=2):  # type: ignore[call-overload]
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple,
        a_dtype,
        b_dtype,
        sf_dtype,
        sfa_smem_layout,
        sfb_smem_layout,
        epi_tile: tuple,
        c_dtype,
        smem_capacity: int,
        occupancy: int,
        epi_stage_override: Optional[int] = None,
    ) -> tuple:
        epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
            tile_shape_mnk[0] // epi_tile[0]
        )

        computed_epi_stage = min(epi_stage_max, 2)
        if epi_stage_override is None:
            epi_stage = computed_epi_stage
        else:
            assert 1 <= epi_stage_override <= epi_stage_max, (
                f"epi_stage={epi_stage_override} exceeds legal max {epi_stage_max} "
                f"for tile_shape_mnk={tile_shape_mnk}, epi_tile={epi_tile}"
            )
            epi_stage = epi_stage_override
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        sf_bytes_per_stage = (
            cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
            + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // (ab_bytes_per_stage + sf_bytes_per_stage)

        ab_stage = max(1, min(ab_stage, 6))
        return ab_stage, epi_stage, epi_stage_max

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        sfa_smem_layout_staged = sm120_make_smem_layout_sfa_cf(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )
        sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c,
        tile_shape_mnk: tuple,
        max_active_clusters,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ) -> tuple:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)


        eff_swizzle = swizzle_size
        try:
            divisor_tile_cnt = int(num_ctas_mnl[1] if raster_along_m else num_ctas_mnl[0])
            while eff_swizzle > 1 and (divisor_tile_cnt % eff_swizzle) != 0:
                eff_swizzle //= 2
        except (TypeError, ValueError):
            pass  # dynamic shape: keep requested swizzle

        try:
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl,
                cluster_shape_mnl,
                swizzle_size=eff_swizzle,
                raster_along_m=raster_along_m,
            )
            print(
                f"Tile scheduler: swizzle_size={eff_swizzle}, "
                f"raster_along_m={raster_along_m}"
            )
        except TypeError:
            tile_sched_params = utils.PersistentTileSchedulerParams(
                num_ctas_mnl, cluster_shape_mnl
            )
            print(
                "Tile scheduler: installed CuTe DSL has no swizzle kwargs; "
                "using default raster order (L2 swizzle NOT active)"
            )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c,
        epi_smem_layout_staged,
        epi_tile: tuple,
    ) -> tuple:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        smem_tile: tuple,
        mcast_dim: int,
        internal_type=None,
    ) -> tuple:
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
            internal_type=internal_type,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size: int,
        c_dtype,
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        if cluster_shape_mn != (1, 1):
            return False
        # Keep this path restricted to the FP4 tile shapes supported by this kernel.
        if tuple(tile_shape_mnk) not in ((128, 128, 128), (128, 128, 256)):
            return False
        # The current target only supports FP4 (MmaMXF4NVF4Op)
        if ab_dtype != cutlass.Float4E2M1FN:
            return False
        if sf_vec_size != 16:
            return False
        if sf_dtype != cutlass.Float8E4M3FN:
            return False
        if c_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        # A must be K-major, B must be K-major
        if a_major != "k" or b_major != "k":
            return False
        # Alignment: K must be divisible by the CTA K tile.
        tile_m, tile_n, tile_k = tile_shape_mnk
        if k % tile_k != 0:
            return False

        sfa_tile_m = max(128, ((tile_m + 127) // 128) * 128)
        sfb_tile_n = max(128, ((tile_n + 127) // 128) * 128)
        ab_bytes = (tile_m * tile_k + tile_n * tile_k) // 2
        sf_bytes = (sfa_tile_m // 128) * 4 * 128 + (sfb_tile_n // 128) * 4 * 128

        epi_bytes = min(128, tile_m) * 32 * 2 * 2
        mbar_bytes = 1024
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        if ab_bytes + sf_bytes + epi_bytes + mbar_bytes > smem_capacity:
            return False
        return True

def ceil_div(a, b):
    return (a + b - 1) // b


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))

    sf_ref_tensor = cute.make_tensor(
        sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k))
    )
    sf_mma_tensor = cute.make_tensor(sf_mma_ptr, cute_layout)

    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]



def _reorder_sf_mkl_to_tile_atom_torch(
    sf_mkl: torch.Tensor, sf_vec_size: int = 16, conflict_free: bool = False
):

    mn, sf_k, l = sf_mkl.shape


    device = sf_mkl.device
    out = torch.empty((mn * sf_k * l,), dtype=sf_mkl.dtype, device=device)

    mn_idx = torch.arange(mn, device=device, dtype=torch.long)[:, None, None]
    sfk_idx = torch.arange(sf_k, device=device, dtype=torch.long)[None, :, None]
    l_idx = torch.arange(l, device=device, dtype=torch.long)[None, None, :]

    atom_m0 = mn_idx % 32
    atom_m1 = (mn_idx // 32) % 4
    rest_m = mn_idx // 128

    atom_k1 = sfk_idx % 4
    rest_k = sfk_idx // 4

    rest_m_extent = mn // 128
    rest_k_extent = sf_k // 4
    atom_cosize = 32 * 4 * 4  

    if conflict_free:
        m0_stride, m1_stride = 4, 32 * 4
    else:
        m0_stride, m1_stride = 16, 4
    physical_offset = (
        atom_m0 * m0_stride
        + atom_m1 * m1_stride
        + atom_k1
        + atom_cosize * (rest_k + rest_k_extent * (rest_m + rest_m_extent * l_idx))
    )

    out[physical_offset.reshape(-1)] = sf_mkl.reshape(-1)
    return out.contiguous()



def create_and_reorder_scale_factor_tensor(
    l, mn, k, sf_vec_size, sf_dtype, torch_tensor, conflict_free=False
):

    torch_tensor = torch_tensor.contiguous()
    sf_k = ceil_div(k, sf_vec_size)
    assert torch_tensor.shape == (mn, sf_k, l), (
        f"expected logical SF shape {(mn, sf_k, l)}, got {tuple(torch_tensor.shape)}"
    )
    assert torch_tensor.dtype == cutlass_torch.dtype(sf_dtype), (
        f"expected SF dtype {cutlass_torch.dtype(sf_dtype)}, got {torch_tensor.dtype}"
    )
    return _reorder_sf_mkl_to_tile_atom_torch(
        torch_tensor, sf_vec_size, conflict_free
    )


def scaled_mm(
    gemm_obj: "Sm120Fp4GemmLauncher",
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    stream: cuda.CUstream,
    options: str = "",
):
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4)

    return cute.compile(
        gemm_obj,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        alpha_ptr,
        stream,
        options=options,
    )


def to_blocked(input_matrix: torch.Tensor):

    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        original_dtype = input_matrix.dtype
        input_float32 = input_matrix.to(torch.float32)
        padded = torch.nn.functional.pad(
            input_float32,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
        if original_dtype != input_float32.dtype:
            padded = padded.to(original_dtype)
    else:
        padded = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def reference_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    mnkl: Tuple[int, int, int, int],
    c_dtype: Type[cutlass.Numeric],
):
    _, _, _, l = mnkl
    c_ref = torch.empty_like(c)
    for l_idx in range(l):
        scale_a = to_blocked(sfa[:, :, l_idx])
        scale_b = to_blocked(sfb[:, :, l_idx])
        a_slice = a[:, :, l_idx].contiguous()
        b_slice = b[:, :, l_idx].contiguous()
        res = torch._scaled_mm(
            a_slice,
            b_slice.transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=cutlass_torch.dtype(c_dtype),
        )
        c_ref[:, :, l_idx] = res
    return c_ref


def construct_cute_pointers(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
):
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfa_ptr = make_ptr(sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    return a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, alpha_ptr


def prepare_tensors(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
):

    m, n, k, l = mnkl
    k_fct = 2 if ab_dtype == cutlass.Float4E2M1FN else 1
    sf_k = ceil_div(k, sf_vec_size)


    sfa = (
        torch.randint(0, 3, (l, m, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype), device="cuda")
    )
    sfb = (
        torch.randint(0, 3, (l, n, sf_k), dtype=torch.uint8)
        .permute(1, 2, 0)
        .to(dtype=cutlass_torch.dtype(sf_dtype), device="cuda")
    )

    a = torch.randint(
        -2, 2, (l, m, k // k_fct), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b = torch.randint(
        -2, 2, (l, n, k // k_fct), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)

    if c_major == "n":
        c = torch.randint(
            -2, 2, (l, m, n), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(1, 2, 0)
    else:
        c = torch.randint(
            -2, 2, (l, n, m), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
        ).permute(2, 1, 0)

    a = a.view(dtype=torch.float4_e2m1fn_x2)
    b = b.view(dtype=torch.float4_e2m1fn_x2)
    c = c.to(dtype=cutlass_torch.dtype(c_dtype))

    return a, b, c, sfa, sfb



class Sm120Fp4GemmLauncher:

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        l: int,
        tile_shape_mnk: Tuple[int, int, int],
        sm_count: int,
        swizzle_size: int = 32,
        epi_tile_m: int = 64,
        epi_tile_n: int = 128,
        ab_stage: Optional[int] = 2,
        epi_stage: Optional[int] = 1,
        raster_along_m: bool = True,
        sf_tma_internal_type: str = "int16",
    ):
        self.m, self.n, self.k, self.l = m, n, k, l
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.sm_count = sm_count
        self.kernel = BlockscaledGemmKernel(
            sf_vec_size=16,
            tile_shape_mnk=tile_shape_mnk,
            cluster_shape_mn=(1, 1),
            swizzle_size=swizzle_size,
            epi_tile_m=epi_tile_m,
            epi_tile_n=epi_tile_n,
            ab_stage=ab_stage,
            epi_stage=epi_stage,
            raster_along_m=raster_along_m,
            sf_tma_internal_type=sf_tma_internal_type,
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        m, n, k, l = self.m, self.n, self.k, self.l
        sf_k = (k + 16 - 1) // 16


        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout((m, k, l), stride=(k, 1, m * k)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout((n, k, l), stride=(k, 1, n * k)),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout((m, n, l), stride=(n, 1, m * n)),
        )

        sfa_tensor = cute.make_tensor(
            sfa_ptr,
            cute.make_layout((m, sf_k, l), stride=(sf_k, 1, m * sf_k)),
        )
        sfb_tensor = cute.make_tensor(
            sfb_ptr,
            cute.make_layout((n, sf_k, l), stride=(sf_k, 1, n * sf_k)),
        )
        alpha_tensor = cute.make_tensor(alpha_ptr, cute.make_layout((1,), stride=(1,)))

        self.kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            alpha_tensor,
            self.sm_count,
            stream,
        )


def run_scaled_mm(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 1,
    iterations: int = 1000,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    swizzle_size: int = 32,
    epi_tile_m: int = 64,
    epi_tile_n: int = 128,
    ab_stage: Optional[int] = 2,
    epi_stage: Optional[int] = 1,
    raster_along_m: bool = True,
    sf_tma_internal_type: str = "int16",
    **kwargs,
):

    print("Running Sm120 Dense BlockScaled GEMM test with:")
    print(f"mnkl: {mnkl}")
    print(f"AB dtype: {ab_dtype}, SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}")
    print(f"C dtype: {c_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mnk}, Cluster Shape (M, N): {cluster_shape_mn}")
    print(f"Scheduler swizzle size: {swizzle_size}, raster_along_m={raster_along_m}")
    print(f"Epilogue tile: ({epi_tile_m}, {epi_tile_n}), epi_stage={epi_stage}")
    print(f"AB stage override: {ab_stage}")
    print(f"SFA/SFB TMA internal_type: {sf_tma_internal_type}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    m, n, k, l = mnkl

    if len(tile_shape_mnk) == 2:
        tile_shape_mnk = (tile_shape_mnk[0], tile_shape_mnk[1], 128)

    if not BlockscaledGemmKernel.can_implement(
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        tile_shape_mnk,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        raise TypeError(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype}, "
            f"{tile_shape_mnk}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, "
            f"{a_major}, {b_major}, {c_major}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    torch_stream = torch.cuda.Stream()
    torch.cuda.set_stream(torch_stream)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    gemm = Sm120Fp4GemmLauncher(
        m=m,
        n=n,
        k=k,
        l=l,
        tile_shape_mnk=tile_shape_mnk,
        sm_count=sm_count,
        swizzle_size=swizzle_size,
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        ab_stage=ab_stage,
        epi_stage=epi_stage,
        raster_along_m=raster_along_m,
        sf_tma_internal_type=sf_tma_internal_type,
    )

    compiled_gemm = scaled_mm(
        gemm,
        ab_dtype,
        c_dtype,
        sf_dtype,
        current_stream,
        options="--opt-level 3",
    )
    # Create Torch Tensors for A, scale factor A, B, scale factor B, C
    a, b, c, sfa, sfb = prepare_tensors(
        mnkl, ab_dtype, sf_dtype, sf_vec_size, c_dtype, a_major, b_major, c_major
    )

    # Reorder scale factor tensors to (32, 4, restM, 4, restK, l) format
    sfa_reordered = create_and_reorder_scale_factor_tensor(
        l, m, k, sf_vec_size, sf_dtype, sfa, conflict_free=True
    )
    sfb_reordered = create_and_reorder_scale_factor_tensor(
        l, n, k, sf_vec_size, sf_dtype, sfb
    )
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")

    # Construct CuTe Pointers
    a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, alpha_ptr = construct_cute_pointers(
        a, b, sfa_reordered, sfb_reordered, c, alpha, ab_dtype, sf_dtype, c_dtype
    )

    # Compute reference result
    if not skip_ref_check:
        # Execute kernel once for reference checking
        compiled_gemm(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, alpha_ptr, current_stream
        )
        torch.cuda.synchronize()
        c_ref = reference_scaled_mm(
            a, b, sfa, sfb, c, mnkl, c_dtype
        )
        torch.testing.assert_close(c.cpu(), c_ref.cpu(), atol=tolerance, rtol=1e-03)

    def generate_inputs():
        a, b, c, sfa, sfb = prepare_tensors(
            mnkl,
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            a_major,
            b_major,
            c_major,
        )
        sfa_reordered = create_and_reorder_scale_factor_tensor(
            l, m, k, sf_vec_size, sf_dtype, sfa, conflict_free=True
        )
        sfb_reordered = create_and_reorder_scale_factor_tensor(
            l, n, k, sf_vec_size, sf_dtype, sfb
        )
        alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
        a_ptr, b_ptr, c_ptr, sfa_ptr, sfb_ptr, alpha_ptr = construct_cute_pointers(
            a, b, sfa_reordered, sfb_reordered, c, alpha, ab_dtype, sf_dtype, c_dtype
        )
        jit_args = cute.testing.JitArguments(
            a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, alpha_ptr, current_stream
        )
        # Keep references to external variables (e.g., Torch tensors when taking a view)
        jit_args.add_to_scope([a, b, sfa_reordered, sfb_reordered, c, alpha])
        return jit_args

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + sfa_reordered.numel() * sfa_reordered.element_size()
            + sfb_reordered.numel() * sfb_reordered.element_size()
            + c.numel() * c.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_inputs,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=True,
    )
    return exec_time


def run(
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    a_major: Literal["m", "k"],
    b_major: Literal["n", "k"],
    c_major: Literal["m", "n"],
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float = 1e-01,
    warmup_iterations: int = 1,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    swizzle_size: int = 32,
    epi_tile_m: int = 64,
    epi_tile_n: int = 128,
    ab_stage: Optional[int] = 2,
    epi_stage: Optional[int] = 1,
    raster_along_m: bool = True,
    sf_tma_internal_type: str = "int16",
    **kwargs,
):

    return run_scaled_mm(
        mnkl,
        ab_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        a_major,
        b_major,
        c_major,
        tile_shape_mnk,
        cluster_shape_mn,
        tolerance,
        warmup_iterations,
        iterations,
        skip_ref_check,
        use_cold_l2,
        swizzle_size=swizzle_size,
        epi_tile_m=epi_tile_m,
        epi_tile_n=epi_tile_n,
        ab_stage=ab_stage,
        epi_stage=epi_stage,
        raster_along_m=raster_along_m,
        sf_tma_internal_type=sf_tma_internal_type,
    )


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Example of Sm120 Dense BlockScaled GEMM."
    )
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(16384, 16384, 16384, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[
            (128, 128, 128),
            (128, 128, 256),
        ],
        default=(128, 128, 128),
        help="CTA tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        default=(1, 1),
        help="Cluster shape (comma-separated); SM120 path currently requires 1,1",
    )
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--swizzle_size",
        type=int,
        default=32,
        help="Tile-raster swizzle width in CTA tiles (L2 locality); 1 disables",
    )
    parser.add_argument(
        "--raster_along_m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persistent scheduler raster direction; default matches best autotune config",
    )
    parser.add_argument(
        "--epi_tile_m",
        type=int,
        default=64,
        help="Epilogue smem subtile size along M; default matches best autotune config",
    )
    parser.add_argument(
        "--epi_tile_n",
        type=int,
        default=128,
        help="Epilogue smem subtile size along N; default matches best autotune config",
    )
    parser.add_argument(
        "--ab_stage",
        type=int,
        default=2,
        help="Override mainloop AB pipeline stages; default matches best autotune config",
    )
    parser.add_argument(
        "--epi_stage",
        type=int,
        default=1,
        help="Override epilogue pipeline stages; default matches best autotune config",
    )
    parser.add_argument(
        "--sf_tma_internal_type",
        choices=["none", "int16"],
        default="int16",
        help="Internal TMA type for SFA/SFB; default matches best autotune config",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=1, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.tile_shape_mnk) != 3:
        parser.error("--tile_shape_mnk must contain exactly 3 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    if args.ab_stage < 0 or args.epi_stage < 0:
        parser.error("--ab_stage and --epi_stage must be >= 0")


    ab_stage = None if args.ab_stage == 0 else args.ab_stage
    epi_stage = None if args.epi_stage == 0 else args.epi_stage

    exec_time = run(
        args.mnkl,
        args.ab_dtype,
        args.sf_dtype,
        16,  # sf_vec_size: NVFP4 block scaling is fixed at 16
        args.c_dtype,
        args.a_major,
        args.b_major,
        args.c_major,
        args.tile_shape_mnk,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        swizzle_size=args.swizzle_size,
        epi_tile_m=args.epi_tile_m,
        epi_tile_n=args.epi_tile_n,
        ab_stage=ab_stage,
        epi_stage=epi_stage,
        raster_along_m=args.raster_along_m,
        sf_tma_internal_type=args.sf_tma_internal_type,
    )
    m, n, k, l = args.mnkl
    flops = 2.0 * m * n * k * l
    tflops = flops / (exec_time * 1e-6) / 1e12
    print(f"Problem size: M={m}, N={n}, K={k}, L={l}")
    print(f"Mean time   : {exec_time:.4f} us")
    print(f"Throughput  : {tflops:.2f} TFLOP/s")
    print("PASS")


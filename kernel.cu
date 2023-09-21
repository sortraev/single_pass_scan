#pragma once
#include "kernel_extras.cu"

// #if ((BLOCK_SIZE) % 32 != 0)
// #error BLOCK_SIZE must be a multiple of 32.
// #endif
// #if ((MAX_CHUNK <= 0))
// #error MAX_CHUNK must be positive.
// #endif

#define FIRST_IN_BLOCK (!threadIdx.x)
#define FIRST_BLOCK    (!blockIdx)

__device__ uint32_t dyn_gic = 0;     // (dyn)amic (g)lobal (i)ndex (c)ounter

template<class OP, uint8_t CHUNK>
__global__
void spas_kernel(uint32_t           N,             // input size in #elements
                 typename OP::ElTp *d_in,          // scan dis!
                 typename OP::ElTp *d_out,         // store res here!
                 typename OP::ElTp *prefixes,
                 typename OP::ElTp *aggregates,
                 uint8_t           *status_flags,
                 uint32_t           num_logical_blocks
                ) {

  typedef typename OP::ElTp ElTp;
  typedef ValFlg<ElTp> FVpair;

  extern __shared__ uint8_t ext_shmem[];
  uint32_t *blockIdx_shmem = (uint32_t*) ext_shmem;
  ElTp     *shmem        = (ElTp*)     ext_shmem;
  FVpair   *fvp_shmem    = (FVpair*)   ext_shmem;

  bool LAST_IN_BLOCK = threadIdx.x + 1 == blockDim.x;
  ElTp chunk[CHUNK];


#if BLOCK_VIRT
  const uint32_t virt_factor = CEIL_DIV(num_logical_blocks, gridDim.x);
  for (int _ = 0; _ < virt_factor; _++)
#endif
  {
  /*
   * step 1: dynamic block indexing
   */
  if (FIRST_IN_BLOCK) {

    uint32_t tmp = atomicAdd(&dyn_gic, 1);
    *blockIdx_shmem = tmp;               // increment dynamic block index
    status_flags[tmp] = flag_X;
                                         // and publish to the rest of the block
#if !(BLOCK_VIRT)
    // when not using virtualization, simply let the last block reset the
    // counter. this is safe since no more blocks are spawned.
    if (tmp == gridDim.x - 1)
      dyn_gic = 0;
#endif
  }

  __syncthreads();
  uint32_t blockIdx = *blockIdx_shmem; // each thread fetches its dynamic blockIdx and stores it locally

#if BLOCK_VIRT
  if (blockIdx >= num_logical_blocks) {
    // TODO: find out how to safely reset dyn_gic.
    if (FIRST_IN_BLOCK)
      dyn_gic = 0;
    return;
  }
#endif

  /*
   * step 2: each thread copies CHUNK elements from global to shared memory
   */
  uint32_t global_block_offset = blockIdx * blockDim.x * CHUNK;
  copyFromGlb2ShrMem<OP, CHUNK>(global_block_offset, N, OP::ne(), (ElTp*) d_in, shmem);

  __syncthreads();

  /*
   * step 3: each thread copies CHUNK elements from shared mem into own "chunk" array;
   *         performs sequential scan of this and places own result back into shared mem.
   */
  uint16_t shmem_offset = threadIdx.x * CHUNK;

  // copy from shared memory to private chunk.
  #pragma unroll
  for (uint8_t i = 0; i < CHUNK; i++)
    chunk[i] = shmem[shmem_offset + i];


  // perform in-place inclusive scan of chunk and store result in shared memory.
  ElTp acc = OP::ne();
  #pragma unroll
  for (uint8_t i = 0; i < CHUNK; i++)
    chunk[i] = acc = OP::apply(acc, chunk[i]);


  __syncthreads();
  shmem[threadIdx.x] = acc;

  /*
   * step 4: in-place block level scan of shmem. store result in block_aggregate.
   *         (this value only meaningful for last thread in each block)
   */
  ElTp block_aggregate = scanIncBlock<OP>(shmem, threadIdx.x);

  if (LAST_IN_BLOCK) {
    (FIRST_BLOCK ? prefixes : aggregates)[blockIdx] = block_aggregate;
    __threadfence();
    status_flags[blockIdx] = FIRST_BLOCK; // = 1 = flag_P if first block; else = 0 = flag_A>
  }

  __syncthreads();

  ElTp chunk_exc_prefix = OP::ne();
  if (!FIRST_IN_BLOCK)
    chunk_exc_prefix = shmem[threadIdx.x-1];       // extract chunk prefixes before shared mem is reused.


  /*
   * step 6: decoupled lookback to compute exclusive prefix.
   */
  ElTp block_exc_prefix = OP::ne();
  if (!FIRST_BLOCK) {

    if (threadIdx.x < WARP) { // only first warp in block performs lookback

      int32_t lookback_idx = blockIdx + threadIdx.x - WARP;
      while (1) {

        FVpair my_fvp = FVpair(flag_P, OP::ne());

        // choose whether to read an aggregate or prefix depending on the flag
        if (lookback_idx >= 0) {
          my_fvp.f = status_flags[lookback_idx];

          if (my_fvp.f & flag_P)
            my_fvp.v = prefixes[lookback_idx];
          else if (!my_fvp.f)
            my_fvp.v = aggregates[lookback_idx];
        }

        fvp_shmem[threadIdx.x] = my_fvp;

        scanIncWarp<FVpairOP<OP> >(fvp_shmem, threadIdx.x);

        FVpair warp_scan_res = FVpairOP<OP>::remVolatile(fvp_shmem[WARP-1]);

        if (warp_scan_res.f >= flag_X) continue;

        if (FIRST_IN_BLOCK) block_exc_prefix = OP::apply(block_exc_prefix, warp_scan_res.v);

        if (warp_scan_res.f & flag_P) break;

        lookback_idx -= WARP;
      }
    }

    /*
     * step 7: publish block_exc_prefix to rest of block before
     *         letting LAST_IN_BLOCK publish block prefix.
     */
    if (FIRST_IN_BLOCK)
      *shmem = block_exc_prefix;

    __syncthreads();
    block_exc_prefix = *shmem;
  }


  if (!FIRST_BLOCK && LAST_IN_BLOCK) {
    prefixes[blockIdx] = OP::apply(block_exc_prefix, block_aggregate);
    // __threadfence_block();
    __threadfence();
    status_flags[blockIdx] = flag_P;
  }


  /*
   * step 7: mapping the exclusive prefix and copying back to global mem
   */
  ElTp my_prefix = OP::apply(block_exc_prefix, chunk_exc_prefix);

  __syncthreads();

  // map my_prefix over private chunk and write to shared mem
  #pragma unroll
  for (uint8_t i = 0; i < CHUNK; i++)
    shmem[shmem_offset + i] = OP::apply(my_prefix, chunk[i]);

  __syncthreads();

  copyFromShr2GlbMem<OP, CHUNK>(global_block_offset, N, (ElTp*) d_out, shmem);
  }
}

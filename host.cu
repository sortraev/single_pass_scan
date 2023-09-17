#include "kernel.cu"
#include "utils.cu"
#include "types.h"
#include "kernel_extras.cu"

template <class OP, uint16_t B, bool do_block_virtualization>
int single_pass_scan(uint32_t           N,
                     typename OP::ElTp *d_in,
                     typename OP::ElTp *d_out,
                     int64_t num_blocks_request = -1
                    ) {

  typedef typename OP::ElTp ElTp;
  typedef ValFlg<ElTp>      FVpair;

  const uint16_t chunk_shmem_bound = MAX_SHMEM / (B * sizeof(ElTp));
  const uint8_t  chunk             = MIN(MAX_CHUNK, chunk_shmem_bound);
  const uint32_t elems_per_block   = B * chunk;
  uint32_t shmem_size = max(elems_per_block * sizeof(ElTp),
                            WARP * sizeof(FVpair));

  uint32_t num_logical_blocks  = CEIL_DIV(N, elems_per_block);
  uint32_t num_physical_blocks = num_logical_blocks;
  if (do_block_virtualization && num_blocks_request > 0)
    num_physical_blocks = MIN((uint32_t) num_blocks_request, num_logical_blocks);
  uint32_t virt_factor = CEIL_DIV(num_logical_blocks, num_physical_blocks);

  ElTp     *aggregates, *prefixes;
  uint8_t  *status_flags;
  CUDASSERT(cudaMalloc(&aggregates,   num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&prefixes,     num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&status_flags, num_logical_blocks*sizeof(uint8_t)));
  CUDASSERT(cudaMemset(status_flags, flag_X, num_logical_blocks*sizeof(uint8_t)));


  printf("spas_kernel bench\n"
         "  block virt = %d\n\n"
         "  block size        = %d\n\n"
         "  #logical blocks   = %d\n"
         "  #requested blocks = %d\n"
         "  #physical blocks  = %d\n"
         "  virt factor       = %d\n\n"
         "  chunk      = %d\n"
         "  shmem_size = %d\n"
         "  N          = %d\n",
         do_block_virtualization,
         B,
         num_logical_blocks,
         num_blocks_request,
         num_physical_blocks,
         virt_factor,
         chunk, shmem_size, N);

  spas_kernel<OP, chunk><<<num_physical_blocks, B, shmem_size>>>(
    N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks,
    virt_factor);

  CUDASSERT(cudaFree(aggregates));
  CUDASSERT(cudaFree(prefixes));
  CUDASSERT(cudaFree(status_flags));

  return 0;
}

// template <class OP>
// int single_pass_scan_no_alloc_aux_arrays(
//     uint32_t N,
//     typename OP::ElTp *d_in,
//     typename OP::ElTp *d_out
// ) {
//   typedef typename OP::ElTp ElTp;
//   typedef ValFlg<ElTp>      FVpair;
//
//   const uint16_t chunk_shmem_bound = MAX_SHMEM / (B * sizeof(ElTp));
//   const uint8_t  chunk             = MIN(MAX_CHUNK, chunk_shmem_bound);
//
//   uint32_t num_logical_blocks  = (N + elems_per_block - 1) / elems_per_block;
//   uint32_t num_physical_blocks = num_logical_blocks;
//   if (do_block_virtualization && num_blocks_request > 0)
//     num_physical_blocks = MIN((uint32_t) num_blocks_request,
//                               num_logical_blocks);
//
//   uint32_t array_size = N * sizeof(ElTp);
//
  /*
   * allocate auxiliary arrays
   */
//
//   CUDASSERT(cudaMemset(d_out, 0, array_size));
//
//   ElTp     *aggregates, *prefixes;
//   uint8_t  *status_flags;
//   CUDASSERT(cudaMalloc(&aggregates,   num_physical_blocks*sizeof(ElTp)));
//   CUDASSERT(cudaMalloc(&prefixes,     num_physical_blocks*sizeof(ElTp)));
//   CUDASSERT(cudaMalloc(&status_flags, num_physical_blocks*sizeof(uint32_t)));
//   CUDASSERT(cudaMemset(status_flags, flag_X, num_physical_blocks*sizeof(uint8_t)));
//   uint32_t shared_mem_size = max(elems_per_block * sizeof(ElTp),
//                                  WARP * sizeof(FVpair));
//
//   // printf("(N, B, CHUNK, num_physical_blocks) == (%d, %d, %d, %d)\n", N, B, chunk, num_physical_blocks);
//   // printf("shared mem per thread: %d\n", shared_mem_size / BLOCK_SIZE);
//   // printf("=========================\n\n");
//
  /*
   *  ================== BENCHMARKING ==================
   */
//     cudaEvent_t t_start, t_end;
//     CUDASSERT(cudaEventCreate(&t_start)); CUDASSERT(cudaEventCreate(&t_end));
//
//     // dry run of kernel
//     spas_kernel<OP, chunk><<<num_physical_blocks, B, shared_mem_size>>>
//       (N, d_in, d_out, prefixes, aggregates, status_flags);
//
//     // invoke SPAS kernel RUNS number of times, measuring total execution time
//     CUDASSERT(cudaEventRecord(t_start));
//     for (uint8_t i = 0; i < RUNS; i++) {
//       spas_kernel<OP, chunk><<<num_physical_blocks, B, shared_mem_size>>>
//         (N, d_in, d_out, prefixes, aggregates, status_flags);
//     }
//     CUDASSERT(cudaEventRecord(t_end));
//     CUDASSERT(cudaEventSynchronize(t_end)); CUDASSERT(cudaPeekAtLastError());
//
//
//
//     // get elapsed and report benchmark result
//     float elapsed = get_elapsed(t_start, t_end, RUNS);
//     float GBPerSec = 2*N*sizeof(ElTp)*0.001 / elapsed;
//
//     printf("--  runs in:       %.1lf microseconds\n", elapsed);
//     printf("--  bandwidth:     %.1f GB/sec\n", GBPerSec);
//     // printf("%d, %lf, %d, %d\n", N, GBPerSec, B, chunk);
//   }
//
//
//   CUDASSERT(cudaFree(aggregates));
//   CUDASSERT(cudaFree(prefixes));
//   CUDASSERT(cudaFree(status_flags));
//
//   return 0;
// }

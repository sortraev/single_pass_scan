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
  const uint32_t elems_per_block = B * chunk;

  uint32_t num_logical_blocks  = (N + elems_per_block - 1) / elems_per_block;
  uint32_t num_physical_blocks = num_logical_blocks;
  if (do_block_virtualization && num_blocks_request > 0)
    num_physical_blocks = MIN((uint32_t) num_blocks_request,
                              num_logical_blocks);

  ElTp     *aggregates, *prefixes;
  uint8_t  *status_flags;
  CUDASSERT(cudaMalloc(&aggregates,   num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&prefixes,     num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&status_flags, num_logical_blocks*sizeof(uint8_t)));
  CUDASSERT(cudaMemset(status_flags, flag_X, num_logical_blocks*sizeof(uint8_t)));

  uint32_t shmem_size = max(elems_per_block * sizeof(ElTp),
                            WARP * sizeof(FVpair));

  printf("spas_kernel\n"
         "  block virtualization = %d\n\n"
         "  block size       = %d\n\n"
         "  #logical blocks   = %d\n"
         "  #requested blocks = %d\n"
         "  #physical blocks  = %d\n\n"
         "  chunk      = %d\n"
         "  shmem_size = %d\n"
         "  N = %d\n",
         do_block_virtualization,
         B,
         num_logical_blocks,
         num_blocks_request,
         num_physical_blocks,
         chunk, shmem_size, N);

  spas_kernel<OP, chunk><<<num_physical_blocks, B, shmem_size>>>(
    N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks);

  CUDASSERT(cudaFree(aggregates));
  CUDASSERT(cudaFree(prefixes));
  CUDASSERT(cudaFree(status_flags));

  return 0;
}

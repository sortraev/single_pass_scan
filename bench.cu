#include "host.cu"
#include "utils.cu"
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef RUNS
#define RUNS 200
#endif


#define NO_VIRT (!(BLOCK_VIRT))

#if BLOCK_VIRT_FC
#define BLOCK_VIRT 0
#endif

int main(int argc, char **argv) {

  typedef typename MyFloat::ElTp ElTp;
  typedef ValFlg<ElTp> FVpair;

  int64_t num_blocks_request = -1;
  if (argc == 3)
    num_blocks_request = atoi(argv[2]);
  else if (argc != 2) {
    fprintf(stderr, "Usage: %s <input size> [optional #physical blocks]\n",
            argv[0]);
    return 1;
  }

  uint32_t N = atoi(argv[1]);
  uint32_t alloc_size = N * sizeof(ElTp);

  // device input/output memory, and host memory for the GPU kernel result.
  ElTp *d_in, *d_out;
  CUDASSERT(cudaMalloc(&d_in,  alloc_size));
  CUDASSERT(cudaMalloc(&d_out, alloc_size));

  // init input array.
  init_array<MyFloat, BLOCK_SIZE>(N, d_in);


  // setup kernel parameters.
  const uint16_t chunk_shmem_bound = MAX_SHMEM / (BLOCK_SIZE * sizeof(ElTp));
  const uint8_t  chunk             = MIN(MAX_CHUNK, chunk_shmem_bound);
  const uint32_t elems_per_block   = BLOCK_SIZE * chunk;
  uint32_t shmem_size = max(elems_per_block * sizeof(ElTp),
                            WARP * sizeof(FVpair));

  uint32_t num_logical_blocks  = CEIL_DIV(N, elems_per_block);
  uint32_t num_physical_blocks = num_logical_blocks;
  if (BLOCK_VIRT && num_blocks_request > 0)
    num_physical_blocks = MIN((uint32_t) num_blocks_request, num_logical_blocks);
  uint32_t virt_factor = CEIL_DIV(num_logical_blocks, num_physical_blocks);

  // init auxiliary arrays.
  ElTp     *aggregates, *prefixes;
  uint8_t  *status_flags;
  CUDASSERT(cudaMalloc(&aggregates,   num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&prefixes,     num_logical_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&status_flags, num_logical_blocks*sizeof(uint8_t)));
  CUDASSERT(cudaMemset(status_flags, flag_X, num_logical_blocks*sizeof(uint8_t)));

#if BLOCK_VIRT
  uint32_t *dyn_gic;
  CUDASSERT(cudaMalloc(&dyn_gic, sizeof(uint32_t)));
  CUDASSERT(cudaMemset(dyn_gic, 0, sizeof(uint32_t)));
#endif

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
         BLOCK_VIRT,
         BLOCK_SIZE,
         num_logical_blocks,
         num_blocks_request,
         num_physical_blocks,
         virt_factor,
         chunk, shmem_size, N);

    cudaEvent_t t_start, t_end;
    CUDASSERT(cudaEventCreate(&t_start)); CUDASSERT(cudaEventCreate(&t_end));

    // dry run for warmup
    spas_kernel
      <Add<MyFloat>, chunk>
      <<<num_physical_blocks, BLOCK_SIZE, shmem_size>>>
#if BLOCK_VIRT
      (N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks, dyn_gic);
    cudaMemset(dyn_gic, 0, sizeof(uint32_t));
#else
      (N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks);
#endif

  CUDASSERT(cudaEventRecord(t_start));
#pragma OPTIMIZE OFF
  for (int i = 0; i < RUNS; i++) {
#pragma OPTIMIZE ON

    spas_kernel
      <Add<MyFloat>, chunk>
      <<<num_physical_blocks, BLOCK_SIZE, shmem_size>>>
#if BLOCK_VIRT
      (N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks, dyn_gic);
    cudaMemset(dyn_gic, 0, sizeof(uint32_t));
#else
      (N, d_in, d_out, prefixes, aggregates, status_flags, num_logical_blocks);
#endif
  }
  CUDASSERT(cudaEventRecord(t_end));
  CUDASSERT(cudaEventSynchronize(t_end));
  CUDASSERT(cudaPeekAtLastError());


  // get elapsed and report benchmark result
  float elapsed = get_elapsed(t_start, t_end, RUNS);
  float GBPerSec = 2 * N * sizeof(ElTp) * 0.001 / elapsed;

  printf("--  execution time:  %.1lf microseconds\n", elapsed);
  printf("--  bandwidth:       %.1f GB/sec\n", GBPerSec);
  // printf("%d, %lf, %d, %d\n", N, GBPerSec, B, chunk);

  CUDASSERT(cudaFree(aggregates));
  CUDASSERT(cudaFree(prefixes));
  CUDASSERT(cudaFree(status_flags));

  CUDASSERT(cudaFree(d_in));
  CUDASSERT(cudaFree(d_out));

  return 0;
}

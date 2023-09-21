#include "host.cu"
#include "utils.cu"
#include <stdlib.h>


#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef VALIDATION_RUNS
#define VALIDATION_RUNS 2
#endif

int main(int argc, char **argv) {

  int64_t num_blocks_request = -1;
  if (argc == 3)
    num_blocks_request = atoi(argv[2]);
  else if (argc != 2) {
    fprintf(stderr, "Usage: %s <input size> [optional #physical blocks]\n",
            argv[0]);
    return 1;
  }
  uint32_t N = atoi(argv[1]);
  uint32_t alloc_size = N * sizeof(MyInt::ElTp);

  // device input/output memory, and host memory for the GPU kernel result.
  MyInt::ElTp *d_in, *d_out;
  CUDASSERT(cudaMalloc(&d_in,  alloc_size));
  CUDASSERT(cudaMalloc(&d_out, alloc_size));
  CUDASSERT(cudaMemset(d_out, 0, alloc_size));

  // host mem for the GPU kernel result.
  MyInt::ElTp *h_out = (MyInt::ElTp*) calloc(sizeof(MyInt::ElTp), N);
  assert(h_out != NULL);


  // host input/output memory for the reference program.
  MyInt::ElTp *seq_in, *seq_out;
  assert((seq_in  = (MyInt::ElTp*) malloc(alloc_size)) != NULL);
  assert((seq_out = (MyInt::ElTp*) malloc(alloc_size)) != NULL);

  // init input arrays.
  init_array<MyInt, BLOCK_SIZE>(N, d_in);
  CUDASSERT(cudaMemcpy(seq_in, d_in, alloc_size, cudaMemcpyDeviceToHost));


  // call GPU kernel and copy result to host mem.

  for (int i = 0; i < VALIDATION_RUNS; i++) {
    single_pass_scan
      <Add<MyInt>, BLOCK_SIZE, BLOCK_VIRT>
      (N, d_in, d_out, num_blocks_request,
       i == 0);
  }

  CUDASSERT(cudaMemcpy(h_out, d_out, alloc_size, cudaMemcpyDeviceToHost));

  // call reference program.
  seq_scan<Add<MyInt> >(N, seq_in, seq_out);

  bool result = validate<MyInt>(N, seq_out, h_out);

  CUDASSERT(cudaFree(d_in)); CUDASSERT(cudaFree(d_out));
  free(h_out);
  free(seq_in); free(seq_out);

  return result;
}

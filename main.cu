#include "utils.cu"
#include <vector>


int i32_mul3(int x) {
  return x * 3;
}

float f32_mul_pi(float x) {
  return x * 3.1416;
}


template <class OP, uint16_t B, typename OP::ElTp (*map_f)(typename OP::ElTp)>
int SPAS(uint32_t           N,
         bool               do_validate,
         typename OP::ElTp *d_in,
         typename OP::ElTp *d_out,
         typename OP::ElTp *h_in    = NULL, // h_in, h_out, and seq_out only used when do_validate == true
         typename OP::ElTp *h_out   = NULL,
         typename OP::ElTp *seq_out = NULL) {

  assert(!do_validate || (h_in != NULL && h_out != NULL && seq_out != NULL));

  typedef typename OP::ElTp ElTp;
  typedef ValFlg<ElTp>      FVpair;

  const uint16_t chunk_shmem_bound = (MAX_SHMEM / (B * sizeof(ElTp)));
  const uint8_t  chunk             = MIN(MAX_CHUNK, chunk_shmem_bound);

  uint16_t elems_per_block = B * chunk;
  uint32_t num_blocks = (N + elems_per_block - 1) / elems_per_block;

  uint32_t array_size = N * sizeof(ElTp);

  /*
   * allocate auxiliary arrays
   */

  CUDASSERT(cudaMemset(d_out, 0, array_size));

  ElTp     *aggregates, *prefixes;
  uint8_t  *status_flags;
  CUDASSERT(cudaMalloc(&aggregates,  num_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&prefixes,     num_blocks*sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&status_flags, num_blocks*sizeof(uint32_t)));
  CUDASSERT(cudaMemset(status_flags, flag_X, num_blocks*sizeof(uint8_t)));
  uint32_t shared_mem_size = max(elems_per_block * sizeof(ElTp),
                                 WARP * sizeof(FVpair));

  printf("(N, B, CHUNK, NUM_BLOCKS) == (%d, %d, %d, %d)\n", N, B, chunk, num_blocks);
  printf("shared mem per thread: %d\n", shared_mem_size / BLOCK_SIZE);
  printf("=========================\n\n");

  if (do_validate) {
    /*
     *   =================== VALIDATION ===================
     */

    // perform sequential scan
    seq_scan<OP>(N, h_in, seq_out, map_f);

    // invoke SPAS :D
    for (int i = 0; i < RUNS; i++) {
      spas_kernel<OP, chunk><<<num_blocks, B, shared_mem_size>>>
        (N, d_in, d_out, prefixes, aggregates, status_flags, map_f);
    }

    // copy back result of SPAS and validate
    CUDASSERT(cudaMemcpy(h_out, d_out, array_size, cudaMemcpyDeviceToHost));
    validate<OP>(N, seq_out, h_out);
  }

  else {
    /*
     *  ================== BENCHMARKING ==================
     */
    cudaEvent_t t_start, t_end;
    CUDASSERT(cudaEventCreate(&t_start)); CUDASSERT(cudaEventCreate(&t_end));

    // dry run of kernel
    spas_kernel<OP, chunk><<<num_blocks, B, shared_mem_size>>>
      (N, d_in, d_out, prefixes, aggregates, status_flags, map_f);
    
    // invoke SPAS kernel RUNS number of times, measuring total execution time
    CUDASSERT(cudaEventRecord(t_start));
    for (uint8_t i = 0; i < RUNS; i++) {
      spas_kernel<OP, chunk><<<num_blocks, B, shared_mem_size>>>
        (N, d_in, d_out, prefixes, aggregates, status_flags, map_f);
    }
    CUDASSERT(cudaEventRecord(t_end));
    CUDASSERT(cudaEventSynchronize(t_end)); CUDASSERT(cudaPeekAtLastError());



    // get elapsed and report benchmark result
    float elapsed = get_elapsed(t_start, t_end, RUNS);
    float GBPerSec = 2*N*sizeof(ElTp)*0.001 / elapsed;

    printf("--  runs in:       %.1lf microseconds\n", elapsed);
    printf("--  bandwidth:     %.1f GB/sec\n", GBPerSec);
    // printf("%d, %lf, %d, %d\n", N, GBPerSec, B, chunk);
  }


  CUDASSERT(cudaFree(aggregates));
  CUDASSERT(cudaFree(prefixes));
  CUDASSERT(cudaFree(status_flags));
  return 0;
}


int main() {
  srand(101);

  std::vector<int> test_input_sizes;
  int num;
  while (std::cin >> num && num >= 1 && num <= 1200000000)
    test_input_sizes.push_back(num);

  for (int N : test_input_sizes) {

    void *d_in, *d_out;
    uint32_t array_size = N * sizeof(float);

    CUDASSERT(cudaMalloc(&d_in,  array_size));
    CUDASSERT(cudaMalloc(&d_out, array_size));

    MyInt::ElTp   *int_d_in    = (MyInt::ElTp*)   d_in;   // reuse allocations for faster benchmarking
    MyInt::ElTp   *int_d_out   = (MyInt::ElTp*)   d_out;
    MyFloat::ElTp *float_d_in  = (MyFloat::ElTp*) d_in;
    MyFloat::ElTp *float_d_out = (MyFloat::ElTp*) d_out;

    // if (DO_VALIDATE) {
      /*
       *  VALIDATION
       */
    //   MyInt::ElTp *h_in, *h_out, *seq_out;
    //   assert((h_in    = (MyInt::ElTp*) malloc(array_size)) != NULL);
    //   assert((h_out   = (MyInt::ElTp*) malloc(array_size)) != NULL);
    //   assert((seq_out = (MyInt::ElTp*) malloc(array_size)) != NULL);
    //
    //   init_array<MyInt, BLOCK_SIZE>(N, int_d_in);
    //   CUDASSERT(cudaMemcpy(h_in, int_d_in, array_size, cudaMemcpyDeviceToHost));
    //   CUDASSERT(cudaMemset(int_d_out, 0, array_size));
    //
    //   SPAS<Add<MyInt>, BLOCK_SIZE>(N, true, int_d_in, int_d_out, h_in, h_out, seq_out);
    //
    //   free(h_in);
    //   free(h_out);
    //   free(seq_out);
    // }
    //
    // else
    {
      /*
       *  BENCHMARKING
       */
      printf("----------- Testing N = %10d -----------\n", N);
      init_array<MyFloat, BLOCK_SIZE>(N, float_d_in);    // init array of random floats in device

      printf("NAIVE MEMCPY - ");
      bandwidthMemcpy<MyFloat, BLOCK_SIZE>(N, float_d_in, float_d_out);


      printf("\nSPAS float32 multiplication - ");
      SPAS<Mult<MyFloat>, BLOCK_SIZE, f32_mul_pi>(N, false, float_d_in, float_d_out, NULL, NULL);
      printf("\nSPAS float32 addition - ");
      SPAS<Add<MyFloat>,  BLOCK_SIZE, f32_mul_pi>(N, false, float_d_in, float_d_out, NULL, NULL);

      printf("----------------------------------------------\n\n\n");
    }

    CUDASSERT(cudaFree(d_in)); CUDASSERT(cudaFree(d_out));
  }

}

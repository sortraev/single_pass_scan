#pragma once
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>


/* if hyperparameters not given as flags to nvcc, set them here */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MAX_CHUNK
#define MAX_CHUNK 9
#endif

#define RUNS      1   /* number of runs performed during benchmarking. */
#define MAX_SHMEM 49152 /* upper bound on shared memory. same for both GPUs, but in the
                           future, we should compute this dynamically for portability.  */

#define MIN(x, y) ((x) < (y) ? x : y) /* since we need a compile-time constant min() function to compute CHUNK */

#define flag_A ((uint8_t) 0)
#define flag_P ((uint8_t) 1)
#define flag_X ((uint8_t) 3)

/*
 * cuda function error wrapper
 */
void CUDASSERT(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(code));
    exit(code);
  }
}


/*
 * sequential scan implementation used for validation.
 */
template<class OP>
typename OP::ElTp seq_scan(uint32_t           N,
                           typename OP::ElTp *h_in,
                           typename OP::ElTp *h_out) {

  typename OP::ElTp acc = OP::ne();
  for (uint32_t i = 0; i < N; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);

  return acc;
}

/*
 * given pointers to two OP::ElTp arrays
 * ref and actual, asserts that they are equal.
 */
template<class OP>
bool validate(uint32_t           N,
              typename OP::ElTp *ref,
              typename OP::ElTp *actual) {

  for (uint32_t i = 0; i < N; i++) {
    if (!OP::equals(ref[i], actual[i])) {

      fprintf(stderr, "\nINVALID!! printing next 10 ...\n"
                      "idx      ref     actual       diff\n");
      for (size_t j = i; j < i + 10; j++)
        fprintf(stderr, "%-8d %-12d %-12d %-10d\n",
                j, ref[j], actual[j], ref[j] - actual[j]);

      return false;
    }
  }

  printf("-- VALID\n");
  return true;
}



/* 
 * kernel to initialize random array in device mem
 */
template <class OP> __global__
void init_device_array(uint32_t           N,
                       typename OP::ElTp *d_in) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
    d_in[gid] = OP::get_random(gid); // use thread id as seed for computationally cheap "random" numbers
}

/*
 * initializes an OP::ElTp array in device mem
 */
template <class OP, uint16_t B>
void init_array(uint32_t           N,
                typename OP::ElTp *d_in) {

  const uint32_t num_blocks = (N + B - 1) / B;
  init_device_array<OP><<<num_blocks, B>>>(N, d_in);
}

/*
 * slightly more convenient interface for computing elapsed time with cudaEvents
 */
float get_elapsed(cudaEvent_t t_start,
                  cudaEvent_t t_end,
                  uint32_t    runs) { 
  float elapsed;
  CUDASSERT(cudaEventElapsedTime(&elapsed, t_start, t_end));
  elapsed *= 1000 / runs; // convert to microseconds and compute average
  return elapsed;
}


/*
 * for the purpose of comparing with a "realistic" bandwidth number
 */
template <class OP>
__global__ void naiveMemcpy(uint32_t N,
                            typename OP::ElTp *d_in,
                            typename OP::ElTp *d_out) {

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
    d_out[gid] = d_in[gid];
}

template <class OP, uint16_t B>
int bandwidthMemcpy(const uint32_t N,
                    typename OP::ElTp *d_in,
                    typename OP::ElTp *d_out) {

  typedef typename OP::ElTp ElTp;
  const uint32_t num_blocks = (N + B - 1) / B;
  printf("num_blocks: %d\n", num_blocks);

  // perform dry run of kernel
  naiveMemcpy<OP><<< num_blocks, B >>>(N, d_in, d_out);

  /*
   * run benchmark
   */
  cudaEvent_t t_start, t_end;
  CUDASSERT(cudaEventCreate(&t_start)); CUDASSERT(cudaEventCreate(&t_end));

  CUDASSERT(cudaEventRecord(t_start));
  for (int i = 0; i < RUNS; i++) {
    naiveMemcpy<OP><<< num_blocks, B >>>(N, d_in, d_out);
  }
  CUDASSERT(cudaEventRecord(t_end));

  /*
   * report elapsed
   */
  CUDASSERT(cudaEventSynchronize(t_end));
  float elapsed = get_elapsed(t_start, t_end, RUNS);
  float GBPerSec = 2*N*sizeof(ElTp)*0.001 / elapsed;
  printf("B == %d\n", B);
  printf("--  runs in:       %.1lf microseconds\n", elapsed);
  printf("--  bandwidth:     %.1f GB/sec\n", GBPerSec);
  
  return 0;
}


/*
 * print max_print_len first elements of arr
 * (as long as arr is an int array hehe)
 */
template <class OP>
__device__ __host__
void print_arr(volatile typename OP::ElTp *arr, uint32_t N) {
  int max_print_len = 64;
  N = min(N, max_print_len);
  
  printf("[");
  for (int i = 0; i < N - 1; i++) {
    printf("%d, ", arr[i]);
  }

  if (N > max_print_len)
    printf("...]\n");
  else if (N > 0)
    printf("%d]\n", arr[N - 1]);
  else
    printf("]\n");
}

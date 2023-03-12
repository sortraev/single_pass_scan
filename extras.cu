#pragma once
#include <stdint.h>
#include "types.cuh"

#define lgWARP 5
#define WARP   (1 << lgWARP)


template <class OP>
__device__ __forceinline__
typename OP::ElTp scanIncWarp(volatile typename OP::ElTp *ptr, uint32_t idx) {
  uint8_t lane = idx & 31;

  if (lane == 0) goto end;
  ptr[idx] = OP::apply(ptr[idx -  1], ptr[idx]);

  if (lane < 2) goto end;
  ptr[idx] = OP::apply(ptr[idx -  2], ptr[idx]);

  if (lane < 4) goto end;
  ptr[idx] = OP::apply(ptr[idx -  4], ptr[idx]);

  if (lane < 8) goto end;
  ptr[idx] = OP::apply(ptr[idx -  8], ptr[idx]);

  if (lane < 16) goto end;
  ptr[idx] = OP::apply(ptr[idx - 16], ptr[idx]);

end:
  return OP::remVolatile(ptr[idx]);
}


template <class OP>
__device__ __forceinline__
typename OP::ElTp scanIncWarp_shfl(volatile typename OP::ElTp *ptr, uint32_t idx) {

  typename OP::ElTp my_val = ptr[idx];

  my_val = __shfl_up_sync((uint8_t)  -1, my_val,  1);
  my_val = __shfl_up_sync((uint8_t)  -2, my_val,  2);
  my_val = __shfl_up_sync((uint8_t)  -4, my_val,  4);
  my_val = __shfl_up_sync((uint8_t)  -8, my_val,  8);
  my_val = __shfl_up_sync((uint8_t) -16, my_val, 16);

  ptr[idx] = my_val;

  return my_val;
}


/*
 * block-level inclusive scan, borrowed from handed-out code for weekly 2
 */
template<class OP>
// __device__ inline typename OP::ElTp
__device__  typename OP::ElTp
scanIncBlock(volatile typename OP::ElTp* ptr, uint32_t idx) {
  uint8_t lane   = idx & (WARP-1);
  uint8_t warpid = idx >> lgWARP;

  // perform warp level scan
  typename OP::ElTp res = scanIncWarp<OP>(ptr, idx);
  __syncthreads();

  // place end-of-warp results in the first warp.
  if (lane == (WARP-1))
    ptr[warpid] = res;

  __syncthreads();

  // re-scan first warp
  if (warpid == 0)
    scanIncWarp<OP>(ptr, idx);

  __syncthreads();

  if (warpid > 0)
    res = OP::apply(ptr[warpid-1], res);

  __syncthreads();

  ptr[idx] = res;
  return res;
}


/*
 *  coalesced copy from global to shared mem, borrowed from handed-out code for weekly 2.
 */
template<class OP, uint8_t CHUNK>
__device__ __forceinline__
void copyFromGlb2ShrMem(uint32_t glb_offs,
                        uint32_t N,
                        const typename OP::ElTp    &ne,
                        typename OP::ElTp          *d_inp,
                        volatile typename OP::ElTp *shmem_inp) {
  #pragma unroll
  for (uint8_t i = 0; i < CHUNK; i++) {

    uint16_t loc_ind = threadIdx.x + blockDim.x * i; 
    uint32_t glb_ind = glb_offs + loc_ind;
    typename OP::ElTp elm = ne;
    if (glb_ind < N)
      elm = d_inp[glb_ind];
    shmem_inp[loc_ind] = elm;
  }
}


/*
 *  coalesced copy from shared to global mem, borrowed from handed-out code for weekly 2.
 */
template<class OP, uint8_t CHUNK>
__device__ __forceinline__
void copyFromShr2GlbMem(uint32_t glb_offs,
                        uint32_t N,
                        typename OP::ElTp          *d_out,
                        volatile typename OP::ElTp *shmem_red) {
  #pragma unroll
  for (uint8_t i = 0; i < CHUNK; i++) {

    uint16_t loc_ind = threadIdx.x + blockDim.x * i;

    uint32_t glb_ind = glb_offs + loc_ind;

    if (glb_ind < N)
      d_out[glb_ind] = shmem_red[loc_ind];
  }
}

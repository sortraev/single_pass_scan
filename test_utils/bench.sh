#!/bin/bash

nvcc="nvcc -O3 -o"
executable=spas_main
src=$executable.cu
kernel=$1

Ns="500000 1000000 5000000 10000000 50000000 100000000 200000000 250000000 300000000 350000000"

GPU_04=a00333.science.domain
if [ "$HOSTNAME" = $GPU_04 ]; then
  Ns+=" 500000000 650000000 800000000 950000000"
  num_Ns="14"
  gpu_name="2080ti"
else
  num_Ns="10"
  gpu_name="780ti"
fi

result_dir=./test_results/benchmarks
mkdir -p $result_dir
out_file=$result_dir/$gpu_name-$kernel-benchmarks.res

$nvcc $executable $src -include $kernel.cu $2 $3 -DDO_VALIDATE=false\
  && echo $num_Ns $Ns\
  | ./$executable\
  | tee $out_file

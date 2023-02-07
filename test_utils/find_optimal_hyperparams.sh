#!/bin/bash

nvcc="nvcc -O3 -o"
executable=spas_main
src=$executable.cu
kernel=$1

block_sizes="224 256 448 512 672 768 896 1024"
chunk_sizes="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
Ns="500000 1000000 5000000 10000000 50000000 100000000 200000000 250000000 300000000 350000000"

GPU_04=a00333.science.domain
if [ "$HOSTNAME" = $GPU_04 ]; then
  Ns+=" 500000000 650000000"
  num_Ns="14"
  gpu_name="2080ti"
else
  num_Ns="10"
  gpu_name="780ti"
fi

test_utils_dir=./test_utils
result_dir=./test_results/optimal_hyperparams
out_file=$result_dir/$gpu_name-$kernel-hyperparams.res
tmp_file=./bench_tmpfile.tmp

mkdir -p $result_dir
rm -f $tmp_file

for block_size in $block_sizes; do
  for chunk_size in $chunk_sizes; do

    hyperparams="-DBLOCK_SIZE=$block_size -DMAX_CHUNK=$chunk_size -DDO_VALIDATE=false"

    $nvcc $executable $src -include $kernel.cu $hyperparams

    echo $num_Ns $Ns | ./$executable | tee -a $tmp_file
  done
done

$test_utils_dir/filter_results.py $tmp_file $out_file

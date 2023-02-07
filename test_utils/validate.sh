#!/bin/bash

# we want to validate our program with various block and chunk sizes on both GPU's
block_sizes="32 224 1024"
chunk_sizes="1 9 16"


Ns="1 7 31 53 770 1270 4830 31337 "   # small inputs
Ns+="133773 7422183 "                 # medium-sized inputs
Ns+="290409865 "                      # large inputs

nvcc="nvcc -O3 -o"
executable=spas_main
src=$executable.cu
kernel=$1


result_dir=./test_results/validation
mkdir -p $result_dir 

GPU_04=a00333.science.domain
if [ "$HOSTNAME" = $GPU_04 ]; then
  gpu_name="2080ti"
  Ns+="983721872" # on the 2080ti, we want to test an even larger input size
else
  gpu_name="780ti"
fi

out_file=$result_dir/$gpu_name-$kernel-validation.res
rm -f $out_file


for block_size in $block_sizes; do
  for chunk_size in $chunk_sizes; do

    # compile with given hyperparameters
    hyperparams="-DBLOCK_SIZE=$block_size -DMAX_CHUNK=$chunk_size -DDO_VALIDATE=true"
    $nvcc $executable $src -include $kernel.cu $hyperparams

    for N in $Ns; do

      # run SPAS, piping error messages to out_file
      echo 1 $N | (./$executable > /dev/null) 2>&1 | tee -a $out_file

      # get exit code of SPAS run
      exit_code=${PIPESTATUS[1]}

      # write validation result to out_file
      if [ $exit_code -ne 0 ]; then
        printf ">>> INVALID for (N, B, MAX_CHUNK) == (%s, %s, %s) :(((\n" $N $block_size $chunk_size | tee -a $out_file
      else
       printf "VALID for (N, B, MAX_CHUNK) == (%s, %s, %s) :D:D:D\n" $N $block_size $chunk_size | tee -a $out_file
      fi
    done
  done
done

# enable validation? (only affects the compile rule)
DO_VALIDATE=0


program=main
src=$(program).cu kernel.cu extras.cu types.h utils.cu

test_utils_dir=./test_utils
test_result_dir=./test_results

nvcc=nvcc -O3 -o

compile: $(program)

run: compile
	echo 1 278452836 | ./$(program)

$(program): $(src)
	$(nvcc) $(program) $(program).cu -include kernel.cu $(hyperparams) -arch=compute_35 -Wno-deprecated-gpu-targets

validate: $(src)
	$(test_utils_dir)/validate.sh kernel

bench: $(src)
	$(test_utils_dir)/bench.sh kernel $(hyperparams)

find_optimal_hyperparams: $(src)
	$(test_utils_dir)/find_optimal_hyperparams.sh kernel

clean:
	rm -f $(program) bench_tmpfile.tmp



# set hyperparameters based on which SPAS kernel we are using and
# which GPU we are currently logged onto.
GPU_04=a00333.science.domain
ifeq ($(HOSTNAME),$(GPU_04))                     # optimal hyperparams for 2080ti
  BLOCK_SIZE=512
  MAX_CHUNK=15
else                                             # optimal hyperparams for 780ti
  BLOCK_SIZE=256
  MAX_CHUNK=9
endif

hyperparams=-DBLOCK_SIZE=$(BLOCK_SIZE) -DMAX_CHUNK=$(MAX_CHUNK) -DDO_VALIDATE=$(DO_VALIDATE)

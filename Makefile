# enable validation? (only affects the compile rule)
DO_VALIDATE=0

program=main
src=$(program).cu kernel.cu extras.cu types.cuh utils.cu

test_utils_dir=./test_utils
test_result_dir=./test_results

nvcc=nvcc -O3 -o

compile: $(program)

run: compile
	echo 1 278452836 | ./$(program)

$(program): $(src)
	# $(nvcc) $(program) $(program).cu -include kernel.cu $(hyperparams) -arch=compute_35 -Wno-deprecated-gpu-targets
	$(nvcc) $(program) $(program).cu -include kernel.cu $(hyperparams)

validate: $(src)
	$(test_utils_dir)/validate.sh kernel

bench: $(src)
	$(test_utils_dir)/bench.sh kernel $(hyperparams)

find_optimal_hyperparams: $(src)
	$(test_utils_dir)/find_optimal_hyperparams.sh kernel

clean:
	rm -f $(program) bench_tmpfile.tmp

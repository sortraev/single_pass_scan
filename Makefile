# enable validation? (only affects the compile rule)
DO_VALIDATE=0

program=main
srcs=host.cu kernel.cu kernel_extras.cu utils.cu types.h

test_utils_dir=./test_utils
test_result_dir=./test_results

nvcc_flags=-O3

compile: $(program)

validate: validate.cu $(srcs)
	nvcc $(nvcc_flags) -DBLOCK_VIRT=1 validate.cu -o $@

clean:
	rm -f $(program) bench_tmpfile.tmp

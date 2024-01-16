nvcc coo-block-spmv.cu -o main.out -O3 -gencode arch=compute_86,code=sm_86 -lcusparse
./main.out
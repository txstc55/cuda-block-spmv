nvcc coo-block-spmv-3-by-4.cu -o main_3_4.out -O3 -gencode arch=compute_86,code=sm_86 -lcusparse
./main_3_4.out

nvcc coo-block-spmv-3-by-3.cu -o main_3_3.out -O3 -gencode arch=compute_86,code=sm_86 -lcusparse
./main_3_3.out

nvcc coo-block-spmv-2-by-3.cu -o main_2_3.out -O3 -gencode arch=compute_86,code=sm_86 -lcusparse
./main_2_3.out

nvcc coo-block-spmv-3-by-2.cu -o main_3_2.out -O3 -gencode arch=compute_86,code=sm_86 -lcusparse
./main_3_2.out
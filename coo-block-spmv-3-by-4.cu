#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#define ELEMENT_SIZE 4
#define DIMENSION 3
#define NUM_EXECUTION 100

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CHECK_CUSPARSE(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = call;                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSPARSE Error: " << status << " at " << __FILE__ << ":"   \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

template <typename T>
std::vector<T> readVectorFromFile(const std::string &filename) {
  std::vector<T> vec;
  std::ifstream inFile(filename);
  T value;

  if (inFile.is_open()) {
    while (inFile >> value) {
      vec.push_back(value);
    }
  }

  inFile.close();
  return vec;
}

template <class F> __device__ __host__ inline F __m_min(F a, F b) {
  return a > b ? b : a;
}

__global__ void coo_spmv(const double *values, const int *elements,
                         const int dimension, const int inputSize,
                         const int numMatrices, const double *x, double *y) {
  // performs COO spmv y = Ax + y
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int matrixRowSize = dimension * inputSize;
  const int matrixSize = matrixRowSize * matrixRowSize;
  if (idx < numMatrices * matrixSize) {
    __shared__ int offset; // For example, matrix is 12 by 12, then with 32
                           // threads, then warp 1 will start with row 2 at
                           // element 8, this 4 is the offset
    int matrixID = idx / matrixSize; // which matrix are we working on
    int matrixRow = (idx % matrixSize) /
                    matrixRowSize; // which row of the matrix are we working on
    int matrixCol =
        (idx % matrixSize) %
        matrixRowSize; // which column of the matrix are we working on
    int smallBlockCol =
        matrixCol / dimension; // which column block are we working on
    int smallBlockColOffset =
        matrixCol %
        dimension; // inside this column block, which column are we working on
    int smallBlockRow = matrixRow / dimension;
    int smallBlockRowOffset = matrixRow % dimension;
    int colOffset = idx % matrixRowSize; // offset from one row

    double rdata =
        values[idx] *
        x[elements[matrixID * inputSize + smallBlockCol] * dimension +
          smallBlockColOffset];

    if (threadIdx.x == 0) {
      offset = (matrixRowSize - colOffset);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + matrixRowSize) /
               matrixRowSize; // check if we are the first couple threads that
                              // are working on the first row in the warp
    int landidx = (threadIdx.x - offset) % matrixRowSize;
    if (BRid == 0) {
      landidx = threadIdx.x;
    }

    int warpId = threadIdx.x % 32;

    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mask = __activemask();
    unsigned int mark = __ballot_sync(mask, bBoundary);
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (warpId + 1)), 31 - warpId);

    for (int iter = 1; iter < matrixRowSize; iter <<= 1) {
      double tmp = __shfl_down_sync(mask, rdata, iter);
      if (interval >= iter)
        rdata += tmp;
    }

    // Only the thread that is the boundary (leader) writes the cpuResult
    if (bBoundary) {
      atomicAdd(&y[elements[matrixID * inputSize + smallBlockRow] * dimension +
                   smallBlockRowOffset],
                rdata);
    }
  }
}

__global__ void coo_spmv_segment(const double *values, const int *elements,
                                 const int dimension, const int inputSize,
                                 const int numMatrices, const double *x,
                                 double *y) {
  // performs COO spmv y = Ax + y
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx *= dimension;
  const int matrixRowSize = dimension * inputSize;
  const int matrixSize = matrixRowSize * matrixRowSize;
  if (idx < numMatrices * matrixSize) {
    __shared__ int offset; // For example, matrix is 12 by 12, then with 32
                           // threads, then warp 1 will start with row 2 at
                           // element 8, this 4 is the offset
    int matrixID = idx / matrixSize; // which matrix are we working on
    int matrixRow = (idx % matrixSize) /
                    matrixRowSize; // which row of the matrix are we working on
    int matrixCol =
        (idx % matrixSize) %
        matrixRowSize; // which column of the matrix are we working on
    int smallBlockCol =
        matrixCol / dimension; // which column block are we working on
    int smallBlockColOffset =
        matrixCol %
        dimension; // inside this column block, which column are we working on
    int smallBlockRow = matrixRow / dimension;
    int smallBlockRowOffset = matrixRow % dimension;
    int colOffset = idx % matrixRowSize; // offset from one row

    double rdata = 0.0;
    for (int i = 0; i < dimension; i++) {
      rdata += values[idx + i] *
               x[elements[matrixID * inputSize + smallBlockCol] * dimension +
                 smallBlockColOffset + i];
    }

    if (threadIdx.x == 0) {
      offset = (matrixRowSize - colOffset);
    }
    __syncthreads();

    int BRid = (threadIdx.x * dimension - offset + matrixRowSize) /
               matrixRowSize; // check if we are the first couple threads that
                              // are working on the first row in the warp
    int landidx = (threadIdx.x * dimension - offset) % matrixRowSize;
    if (BRid == 0 && threadIdx.x < matrixRowSize) {
      landidx = threadIdx.x;
    }

    int warpId = threadIdx.x % 32;

    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mask = __activemask();
    unsigned int mark = __ballot_sync(mask, bBoundary);
    mark = __brev(mark);
    unsigned int interval = min(__clz(mark << (warpId + 1)), 31 - warpId);

    for (int iter = 1; iter < matrixRowSize; iter <<= 1) {
      double tmp = __shfl_down_sync(mask, rdata, iter);
      if (interval >= iter)
        rdata += tmp;
    }

    // Only the thread that is the boundary (leader) writes the cpuResult
    if (bBoundary) {
      atomicAdd(&y[elements[matrixID * inputSize + smallBlockRow] * dimension +
                   smallBlockRowOffset],
                rdata);
    }
  }
}

int main() {
  std::vector<int> elements = readVectorFromFile<int>("elements.txt");
  std::vector<double> values =
      readVectorFromFile<double>("additionalHessianResults.txt");
  int MATRIX_SIZE =
      *std::max_element(elements.begin(), elements.end()) * DIMENSION +
      DIMENSION;
  printf("Matrix size: %d\n", MATRIX_SIZE);

  // Expanded matrix calculations
  std::vector<int> expanded_rows(values.size());
  std::vector<int> expanded_cols(values.size());
  int count = 0;

  const int matrixSize = ELEMENT_SIZE * DIMENSION * ELEMENT_SIZE * DIMENSION;
  const int rowSize = ELEMENT_SIZE * DIMENSION;
  for (int idx = 0; idx < values.size(); idx++) {
    int matrixID = idx / matrixSize;
    int matrixRow = (idx % matrixSize) / rowSize;
    int matrixCol = (idx % matrixSize) % rowSize;
    int smallBlockRow = matrixRow / DIMENSION;
    int smallBlockCol = matrixCol / DIMENSION;
    int smallBlockRowOffset = matrixRow % DIMENSION;
    int smallBlockColOffset = matrixCol % DIMENSION;

    std::vector<int> localElements(ELEMENT_SIZE);
    for (int i = 0; i < ELEMENT_SIZE; i++) {
      localElements[i] = elements[matrixID * ELEMENT_SIZE + i];
    }

    int expandedRow =
        localElements[smallBlockRow] * DIMENSION + smallBlockRowOffset;
    int expandedCol =
        localElements[smallBlockCol] * DIMENSION + smallBlockColOffset;

    expanded_rows[count] = expandedRow;
    expanded_cols[count] = expandedCol;

    count++;
    if (expandedRow >= MATRIX_SIZE || expandedCol >= MATRIX_SIZE) {
      printf("Invalid expanded row/col: %d, %d\n", expandedRow, expandedCol);
      exit(EXIT_FAILURE);
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  // read random x from file
  std::vector<double> h_x = readVectorFromFile<double>("fake_x.txt");
  // Allocate memory for vector x and y, and the random states
  double *d_x, *d_y_segment, *d_y;
  curandState *d_states;
  CHECK_CUDA(cudaMalloc(&d_x, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_y_segment, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_y, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMemset(d_y_segment, 0, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMemset(d_y, 0, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_states, MATRIX_SIZE * sizeof(curandState)));
  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), MATRIX_SIZE * sizeof(double),
                        cudaMemcpyHostToDevice));

  // get the true result from file
  std::vector<double> h_y(MATRIX_SIZE, 0.0);
  std::vector<double> trueResult = readVectorFromFile<double>("fake_y.txt");

  // let's do spmv
  std::vector<double> cpuResult(MATRIX_SIZE, 0.0);
  for (int i = 0; i < values.size(); i++) {
    int row = expanded_rows[i];
    int col = expanded_cols[i];
    cpuResult[row] += values[i] * h_x[col];
  }


  // Allocate and copy COO format data
  int *d_expanded_rows, *d_expanded_cols, *d_elements;
  double *d_values;
  CHECK_CUDA(cudaMalloc((void **)&d_expanded_rows,
                        expanded_rows.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_expanded_cols,
                        expanded_cols.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_values, values.size() * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&d_elements, elements.size() * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_expanded_rows, expanded_rows.data(),
                        expanded_rows.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_expanded_cols, expanded_cols.data(),
                        expanded_cols.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_values, values.data(), values.size() * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_elements, elements.data(),
                        elements.size() * sizeof(int), cudaMemcpyHostToDevice));

  // ===========================================================
  // the segmented methodnvcc coo-block-spmv-3-by-3.cu -o main_3_3.out -O3
  // -gencode arch=compute_86,code=sm_86 -lcusparse
  // ===========================================================
  cudaEventRecord(start);
  for (int i = 0; i < NUM_EXECUTION; i++) {
    coo_spmv_segment<<<(values.size() / 3 + 31) / 32, 32>>>(
        d_values, d_elements, DIMENSION, ELEMENT_SIZE,
        elements.size() / ELEMENT_SIZE, d_x, d_y_segment);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time averaged over %d executions for segmented spmv: %f\n",
         NUM_EXECUTION, milliseconds / NUM_EXECUTION);

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y_segment, MATRIX_SIZE * sizeof(double),
                        cudaMemcpyDeviceToHost));


  double error = 0.0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    error += abs(h_y[i] / NUM_EXECUTION - trueResult[i]);
  }
  printf("Error for segmented with true result: %f\n", error);

  // ===========================================================
  // the non segmented method
  // ===========================================================
  cudaEventRecord(start);
  for (int i = 0; i < NUM_EXECUTION; i++) {
    coo_spmv<<<(values.size() + 31) / 32, 32>>>(
        d_values, d_elements, DIMENSION, ELEMENT_SIZE,
        elements.size() / ELEMENT_SIZE, d_x, d_y);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time averaged over %d executions for non segmented spmv: %f\n",
         NUM_EXECUTION, milliseconds / NUM_EXECUTION);

  CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, MATRIX_SIZE * sizeof(double),
                        cudaMemcpyDeviceToHost));


  error = 0.0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    error += abs(h_y[i] / NUM_EXECUTION - trueResult[i]);
  }
  printf("Error for non segmented with true result: %f\n", error);

  // Cleanup
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_expanded_cols));
  CHECK_CUDA(cudaFree(d_expanded_rows));
  CHECK_CUDA(cudaFree(d_elements));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y_segment));
  CHECK_CUDA(cudaFree(d_states));

  return 0;
}

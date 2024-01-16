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
#define NUM_EXECUTION 1

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

__global__ void setup_kernel(curandState *state, unsigned long seed) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__ void generateRandomVector(double *vec, curandState *states,
                                     unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    vec[idx] = curand_uniform_double(&states[idx]);
    vec[idx] = 1.0;
  }
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

    int BRid = (threadIdx.x - offset + matrixRowSize) / matrixRowSize;
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


    // Only the thread that is the boundary (leader) writes the result
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
  // for (int i = 0; i < values.size(); i++) {
  //   values[i] = 1.0;
  // }
  int MATRIX_SIZE =
      *std::max_element(elements.begin(), elements.end()) * DIMENSION +
      DIMENSION;
  printf("Matrix size: %d\n", MATRIX_SIZE);

  // Expanded matrix calculations
  std::vector<int> expanded_rows(values.size());
  std::vector<int> expanded_cols(values.size());
  int count = 0;
  std::map<int, std::map<int, double>> matrixResult;

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

    if (matrixResult.find(expandedRow) == matrixResult.end()) {
      matrixResult[expandedRow] = std::map<int, double>();
    }
    if (matrixResult[expandedRow].find(expandedCol) == matrixResult[expandedRow].end()) {
      matrixResult[expandedRow][expandedCol] = 0.0;
    }
    matrixResult[expandedRow][expandedCol] += values[idx];
    
    expanded_rows[count] = expandedRow;
    expanded_cols[count] = expandedCol;

    count++;
    if (expandedRow >= MATRIX_SIZE || expandedCol >= MATRIX_SIZE) {
      printf("Invalid expanded row/col: %d, %d\n", expandedRow, expandedCol);
      exit(EXIT_FAILURE);
    }
  }

  int selectedRow = 1;
  int selectedCol = 3;
  for (int i = 0; i < DIMENSION; i++){
    for (int j = 0; j < DIMENSION; j++){
      printf("%lf ", matrixResult[selectedRow * DIMENSION + i][selectedCol * DIMENSION + j]);
    }
    printf("\n");
  }
  printf("\n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;

  // Allocate memory for vector x and y, and the random states
  double *d_x, *d_y;
  curandState *d_states;
  CHECK_CUDA(cudaMalloc((void **)&d_x, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMalloc((void **)&d_y, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMemset(d_y, 0, MATRIX_SIZE * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_states, MATRIX_SIZE * sizeof(curandState)));
  // Generate random vector x
  setup_kernel<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_states, 13);
  generateRandomVector<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_x, d_states,
                                                           MATRIX_SIZE);
  CHECK_CUDA(cudaDeviceSynchronize());
  // check the first 10 elements of x
  double *h_x = (double *)malloc(MATRIX_SIZE * sizeof(double));
  CHECK_CUDA(cudaMemcpy(h_x, d_x, MATRIX_SIZE * sizeof(double),
                        cudaMemcpyDeviceToHost));
  

  // let's do spmv
  std::vector<double> result(MATRIX_SIZE, 0.0);
  for (int i = 0; i < values.size(); i++) {
    int row = expanded_rows[i];
    int col = expanded_cols[i];
    result[row] += values[i] * h_x[col];
  }

  // check the first 10 results of result
  for (int i = 0; i < 10; i++) {
    printf("%lf ", result[i]);
  }
  printf("\n");
  

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
  cudaEventRecord(start);
  for (int i = 0; i < NUM_EXECUTION; i++) {

    coo_spmv<<<(values.size() + 31) / 32, 32>>>(
        d_values, d_elements, DIMENSION, ELEMENT_SIZE,
        elements.size() / ELEMENT_SIZE, d_x, d_y);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time: %f\n", milliseconds / NUM_EXECUTION);

  
  // check the first 10 results of d_y
  double *h_y = (double *)malloc(MATRIX_SIZE * sizeof(double));
  CHECK_CUDA(cudaMemcpy(h_y, d_y, MATRIX_SIZE * sizeof(double),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; i++) {
    printf("%lf ", h_y[i]);
  }
  printf("\n");


  double error = 0.0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    error += abs(h_y[i] - result[i]);
  }
  printf("Error: %f\n", error);

  // Cleanup
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_expanded_cols));
  CHECK_CUDA(cudaFree(d_expanded_rows));
  CHECK_CUDA(cudaFree(d_elements));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_states));
  free(h_x);

  return 0;
}

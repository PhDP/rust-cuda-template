#include <cublas_v2.h>
#include "ruda/matrix.h"

float* ruda_mm32(const float* a, const float* b, const int m, const int k, const int n) {
  size_t const a_bytes = m * k * sizeof(float);
  size_t const b_bytes = k * n * sizeof(float);
  size_t const c_bytes = m * n * sizeof(float);
  
  float* c = (float*)malloc(c_bytes);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, a_bytes);
  cudaMalloc(&d_B, b_bytes);
  cudaMalloc(&d_C, c_bytes);

  cudaMemcpy(d_A, a, a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b, b_bytes, cudaMemcpyHostToDevice);

  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m);

  // Destroy the handle
  cublasDestroy(handle);

  cudaMemcpy(c, d_C, c_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return c;
}


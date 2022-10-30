#ifndef RUDA_MM_H_
#define RUDA_MM_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
  \brief Column-major matrix multiplication on the GPU with cublas.

  \param a  m x k matrix.
  \param b  k x n matrix.
  \param m  Number of rows of the matrix 'a'.
  \param k  Number or columns of the matrix 'b' and rows of the matrix 'a'.
  \param n  Number of columns of the matrix 'b'.
  \return   m x n matrix.
 */
float* ruda_mm32(const float* a, const float* b, const int m, const int k, const int n);

#ifdef __cplusplus
}
#endif

#endif


// https://github.com/sheredom/utest.h

#include "ruda/matrix.h"
#include "utest.h"

UTEST_MAIN();

int approx(float x, float y) {
  return x < y? approx(y, x) : x - y < 0.1;
}

UTEST(rudatest, matrixmultiplication) {
  const int m = 4;
  const int k = 2;
  const int n = 3;

  float* matrix_a = (float*)malloc(m * k * sizeof(float));
  float* matrix_b = (float*)malloc(k * n * sizeof(float));

  matrix_a[0] = 0.5f;
  matrix_a[1] = 2.7f;
  matrix_a[2] = -0.3f;
  matrix_a[3] = 8.1f;
  matrix_a[4] = 42.59f;
  matrix_a[5] = 16.75f;
  matrix_a[6] = -128.0f;
  matrix_a[7] = 3.99f;
  
  matrix_b[0] = 5.25;
  matrix_b[1] = -14.81;
  matrix_b[2] = 12.01;
  matrix_b[3] = 14.6;
  matrix_b[4] = -189.5;
  matrix_b[5] = 0.13;

/*
import numpy as np
a = np.array([[0.5, 42.59], [2.7, 16.75], [-0.3, -128], [8.1, 3.99]])
b = np.array([[5.25, 12.01, -189.5], [-14.81, 14.6, 0.13]])
c = np.matmul(a, b) 
print(c)
*/

  float* matrix_c = ruda_mm32(matrix_a, matrix_b, m, k, n);

  ASSERT_TRUE(approx(matrix_c[0], -628.1329));
  ASSERT_TRUE(approx(matrix_c[1], -233.8925));
  ASSERT_TRUE(approx(matrix_c[2], 1894.105));
  ASSERT_TRUE(approx(matrix_c[3], -16.5669));
  ASSERT_TRUE(approx(matrix_c[4], 627.819));
  ASSERT_TRUE(approx(matrix_c[5], 276.977));
  ASSERT_TRUE(approx(matrix_c[6], -1872.403));
  ASSERT_TRUE(approx(matrix_c[7], 155.535));
  ASSERT_TRUE(approx(matrix_c[8], -89.2133));
  ASSERT_TRUE(approx(matrix_c[9], -509.4725));
  ASSERT_TRUE(approx(matrix_c[10], 40.21));
  ASSERT_TRUE(approx(matrix_c[11], -1534.4313));

  free(matrix_a);
  free(matrix_b);
  free(matrix_c);
}


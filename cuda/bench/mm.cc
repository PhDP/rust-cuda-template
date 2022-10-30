#include "benchmark/benchmark.h"
#include "ruda/matrix.h"
#include <cstdlib>

float* naive_mm(float *a, float *b, int m, int k, int n) {
  float *c = (float*)std::malloc(m * n * sizeof(float));
  float *c_ptr = c;
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      *c_ptr = 0.0;
      for (size_t i = 0; i < k; ++i) {
        *c_ptr += a[row + i * m] * b[i + col * k];
      }
      ++c_ptr;
    }
  }
  return c;
}

float* fill(size_t n, float f) {
  float* x = (float*)std::malloc(n * sizeof(float));
  for (size_t i = 0; i < n; ++i) x[i] = f;
  return x;
}

static void BM_NaiveMatrixMult(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    float* x = fill(state.range(0) * state.range(0), 1.5);
    float* y = fill(state.range(0) * state.range(0), 1.5);
    state.ResumeTiming();
    float* z = naive_mm(x, y, state.range(0), state.range(0), state.range(0));
    state.PauseTiming();
    std::free(x);
    std::free(y);
    std::free(z);
    state.ResumeTiming();
  }
}
BENCHMARK(BM_NaiveMatrixMult)
    ->Args({100})
    ->Args({300})
    ->Args({500})
    ->Args({700})
    ->Args({900})
    ->Args({1000});

static void BM_CudaMatrixMult(benchmark::State& state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    float* x = fill(state.range(0) * state.range(0), 1.5);
    float* y = fill(state.range(0) * state.range(0), 1.5);
    state.ResumeTiming();
    float* z = ruda_mm32(x, y, state.range(0), state.range(0), state.range(0));
    state.PauseTiming();
    std::free(x);
    std::free(y);
    std::free(z);
    state.ResumeTiming();
  }
}
BENCHMARK(BM_CudaMatrixMult)
    ->Args({100})
    ->Args({300})
    ->Args({500})
    ->Args({700})
    ->Args({900})
    ->Args({1000});

BENCHMARK_MAIN();

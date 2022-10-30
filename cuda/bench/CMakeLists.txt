set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

add_subdirectory(benchmark)
include_directories(benchmark/include)

set(bench_cc
  mm.cc
)

add_executable(bench_ruda ${bench_cc})

target_link_libraries(bench_ruda
  benchmark
  ruda
  ${MATH_LIBRARIES}
)

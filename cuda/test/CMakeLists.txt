add_subdirectory(gtest)

include_directories(./gtest/googletest/include)

set(test_src
  run_all.cc
  matrix_mult_test.cc
)

add_executable(test_ruda ${test_src})

target_link_libraries(test_ruda
  gtest
  ruda
  ${MATH_LIBRARIES}
)

add_test(
  NAME tests
  COMMAND test_ruda
)


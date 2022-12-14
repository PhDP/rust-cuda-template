cmake_minimum_required(VERSION 3.2)
project(ruda)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

option(RUDA_TEST      "Tests with Google Test"            OFF)
option(RUDA_BENCH     "Bechmarks with Google Benchmark"   OFF)
option(RUDA_STATIC    "Make a static library"             ON)

if (RUDA_STATIC)
  set(RUDA_LIB_TYPE STATIC)
else ()
  set(RUDA_LIB_TYPE SHARED)
endif ()

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(Math)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(. include test ${CUDA_INCLUDE_DIRS})

set(ruda_src matrix.cu)

cuda_add_library(ruda ${RUDA_LIB_TYPE} ${ruda_src})

target_link_libraries(ruda
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBRARIES}
)

install(TARGETS ruda LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ./include/ruda DESTINATION include)

if (RUDA_TEST)
  enable_testing()
  add_subdirectory(test)
endif ()

if (RUDA_BENCH)
  add_subdirectory(bench)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests          : ${RUDA_TEST}")
message(STATUS "  Build benchmarks     : ${RUDA_BENCH}")
message(STATUS "  Library name         : ruda")
message(STATUS "  Library type         : ${RUDA_LIB_TYPE}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "")


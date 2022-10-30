# rust-cuda-template

Small example of how to compile & use cuda kernels with Rust. It requires CUDA
11+ (hence the use of the C++17 standard) but these parts can easily be
removed.

I've added CUDA-specific tests/benchmarks using Google Test and Google
Benchmark so the kernels can be tested outside of the Rust code. Since they use
git submodules you may want to close this repository with:

    $ git clone --recursive git@github.com:PhDP/rust-cuda-template.git



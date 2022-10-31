# rust-cuda-template

Small example of how to compile & use cuda kernels with Rust.

'src/lib.rs' shows how to use the kernel in 'cuda/matrix.cu'.

To compile the CUDA code and run the test:

    $ cargo test

I've added CUDA-specific tests/benchmarks using Google Test and Google
Benchmark so the kernels can be tested outside of the Rust code. Since they use
git submodules you may want to close this repository with:

    $ git clone --recursive git@github.com:PhDP/rust-cuda-template.git

## License

You are free to clone this repository and adopt any license you
wish to adopt for your code.


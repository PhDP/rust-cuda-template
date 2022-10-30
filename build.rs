fn main() {
    cc::Build::new()
        .out_dir(std::path::Path::new("lib"))
        .cuda(true)
        .include("cuda/include")
        .include("cuda")
        .flag("-cudart=shared")
        .flag("-O2")
        .file("cuda/matrix.cu")
        .compile("libruda.a");
 
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cuda");
}


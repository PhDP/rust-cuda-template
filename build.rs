fn main() {
    cc::Build::new()
        .out_dir(std::path::Path::new("lib"))
        .cuda(true)
        .flag("-cudart=shared")
        .include("ruda/include")
        .include("ruda")
        .file("ruda/matrix.cu")
        .flag("-O2")
        .compile("libruda.a");
 
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!("cargo:rustc-link-search=native=lib");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cuda");
}


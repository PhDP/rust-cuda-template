fn main() {
    let mut cfg = cc::Build::new();
    cfg.cuda(true);
    cfg.include("ruda/include")
        .include("ruda")
        .file("ruda/matrix.cu")
        .out_dir(std::path::Path::new("lib"))
        .flag("-O2")
        .compile("libruda.a");
 
    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-search=lib");
    println!("cargo:rustc-link-lib=static=ruda");
}

fn main() {
    const RABITQ_INCLUDE: &str = "../third_party/rabitq/include";

    // Tell cargo to look for shared libraries in the specified directory
    // println!("cargo:rustc-link-search=/opt/intel/oneapi/mkl/latest/lib/intel64");
    // println!("cargo:rustc-link-lib=mkl_intel_lp64");
    // println!("cargo:rustc-link-lib=mkl_sequential");
    // println!("cargo:rustc-link-lib=mkl_core");
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=gomp");
    // lapack library for matrix operations
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=blas");
    println!("cargo:rustc-link-lib=cblas");

    // build rabitq_functions.cpp
    cc::Build::new()
        .cpp(true)
        .file("rabitq_functions.cpp")
        .include(".")
        .include(RABITQ_INCLUDE)
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-fopenmp")
        .flag("-march=native")
        .flag("-mavx512f")
        .flag("-mavx512vl")
        .flag("-mavx512bw")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-msse2")
        .flag("-ftree-vectorize")
        .cargo_warnings(false)
        .compile("rabitq_functions");

    // Re-run if C++ files change
    println!("cargo:rerun-if-changed=rabitq_functions.cpp");
    println!("cargo:rerun-if-changed=rabitq_functions.h");
    println!("cargo:rerun-if-changed={RABITQ_INCLUDE}");
}

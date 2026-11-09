#!/usr/bin/env bash
# Source this file to use this checkout and its locally installed libraries.
export SHEAR_ROOT="${SHEAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export SHEAR_DATA_ROOT="${SHEAR_DATA_ROOT:-$SHEAR_ROOT/data}"
export SHEAR_OUTPUT_ROOT="${SHEAR_OUTPUT_ROOT:-$SHEAR_ROOT/runs}"
export DATA_ROOT="${DATA_ROOT:-$SHEAR_DATA_ROOT}"
shear_local_lib="$SHEAR_ROOT/.local-deps/root/usr/lib/x86_64-linux-gnu/openblas-pthread"
export LD_LIBRARY_PATH="$shear_local_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$shear_local_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PATH="$HOME/.cargo/bin:$PATH"
# Host build scripts must not execute AVX-512 on this AVX2 machine.
export RUSTFLAGS="${RUSTFLAGS:--C target-feature=+avx2 -L /usr/lib/x86_64-linux-gnu -l lapacke -l lapack -l blas}"
export CXXFLAGS="${CXXFLAGS:--mavx512vpopcntdq}"
unset shear_local_lib

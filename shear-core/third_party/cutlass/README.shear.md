# CUTLASS headers for SQ4+ leaf MMA

Pinned NVIDIA CUTLASS v3.9.2; commit and upstream archive hash are in UPSTREAM.json. These 93 headers are the transitive literal-include dependency closure of cutlass/arch/mma_sm80.h, copied without modifications. CUDA standard-library headers come from the existing CUDA toolkit. See LICENSE.txt for redistribution terms.

Only the opt-in SQ4+ leaf MMA flag uses these headers. The Rust build records this crate-relative include directory for NVRTC; retain the headers at that location when replaying the binary. NVRTC uses C++17 and --device-as-default-execution-space for the unannotated upstream utility functions. The baseline does not include CUTLASS. SHEAR's indexing, triangle traversal, distance restoration and top-k remain in compute_backend.rs; no CUTLASS SYRK/grouped scheduler is wired in by this experiment.

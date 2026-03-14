// SIMD_Auto.rs — Design document for SIMD dispatch strategy
//
// Four targets. Three compile-time, one runtime:
//
// BINARY          HOW                              WHO
// ─────────────────────────────────────────────────────────────
// avx512          -C target-cpu=native             Production server (known AVX-512)
// avx2            -C target-feature=+avx2,+fma     Production laptop (known AVX2)
// auto            default, no flags                CI, pip install, "just works"
// arm             --target aarch64-*               Mac, Graviton, RPi
//
// The `auto` binary is what PR #102 builds:
//   simd.rs uses is_x86_feature_detected! per function.
//   Runtime dispatch: AVX-512 → AVX2 → scalar.
//   Works everywhere. Zero config.
//
// The other three are just build flags. No code changes needed.
// The same source compiles all four. When target_feature is set at
// compile time, LLVM inlines through the is_x86_feature_detected!
// branch — the check becomes a constant true, the dead branch is
// eliminated. Zero overhead on the dedicated binaries too.
//
// Architecture:
//
//   simd_avx512.rs  — AVX-512 wrapper types (F32x16, F64x8, U8x64, etc.)
//                     + inherent methods (splat, from_slice, reduce_sum, etc.)
//
//   simd_avx2.rs    — AVX2 BLAS-1 functions (dot_f32, axpy_f32, etc.)
//                     Uses f32x8/f64x4 from simd_avx512.rs
//
//   scalar_fns.rs   — Scalar fallback for every function
//                     Runs on ANY architecture. LLVM may auto-vectorize.
//
//   simd.rs         — PUBLIC API. Runtime dispatch per function:
//                     if avx512f → inline AVX-512 impl
//                     else if avx2+fma → call simd_avx2::*
//                     else → scalar fallback
//                     Also: hamming, popcount, dot_i8 with fn-pointer dispatch.
//
//   simd_isa.rs     — Isa trait: bridge stable simd_avx512 ↔ nightly std::simd
//
// Contract:
//   • cargo test --workspace passes with NO flags (auto binary)
//   • No Cargo feature flags needed for ISA selection
//   • avx512=[] and avx2=[] kept as empty no-ops for archive crate compat
//   • #[cfg(target_arch = "x86_64")] gates all x86 intrinsics
//   • Scalar fallback compiles on all targets (ARM, WASM, etc.)

// SIMD_Auto.rs — Design document for SIMD dispatch strategy
//
// Four targets. Three compile-time, one runtime:
//
// BINARY          HOW                              WHO
// ─────────────────────────────────────────────────────────────
// auto            default, no flags                CI, pip install, "just works"
// avx512          -C target-feature=+avx512f       Production server (known AVX-512)
// avx2            -C target-feature=+avx2,+fma     Production laptop (known AVX2)
// arm             --target aarch64-*               Mac, Graviton, RPi
//
// ─── auto binary (PR #102) ───────────────────────────────────────────
//
// Compiled WITHOUT target flags. ALL branches stay alive.
// simd.rs uses is_x86_feature_detected! per function call.
// Runtime dispatch: AVX-512 → AVX2 → scalar.
// Cost: one cached atomic load per call. Negligible.
// LLVM does NOT eliminate any branch — they must all stay live
// so the binary works on any CPU.
//
// ─── dedicated binaries ──────────────────────────────────────────────
//
// Compiled WITH target flags. #[cfg(target_feature)] gates exclude
// the other modules entirely at compile time. Different mechanism
// from the auto binary — not LLVM folding runtime checks, but
// compile-time gates that prevent other ISA code from being emitted.
//
//   avx512: -C target-feature=+avx512f → only AVX-512 code
//   avx2:   -C target-feature=+avx2    → only AVX2 code
//   arm:    --target aarch64            → only scalar code
//
// ─── Architecture ────────────────────────────────────────────────────
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
// ─── Contract ────────────────────────────────────────────────────────
//
//   • cargo test --workspace passes with NO flags (auto binary)
//   • No Cargo feature flags needed for ISA selection
//   • avx512=[] and avx2=[] kept as empty no-ops for archive crate compat
//   • #[cfg(target_arch = "x86_64")] gates all x86 intrinsics
//   • Scalar fallback compiles on all targets (ARM, WASM, etc.)
//   • Auto binary: ALL branches live, runtime picks best ISA
//   • Dedicated binaries: compile-time #[cfg(target_feature)] excludes others

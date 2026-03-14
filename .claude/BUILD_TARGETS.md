# BUILD_TARGETS.md

## Four Build Targets

```
TARGET              BUILD COMMAND                          DISPATCH        USE CASE
───────────────────────────────────────────────────────────────────────────────────────
rustynum-universal  cargo build                            runtime         CI, pip install, "just works"
rustynum-avx512     RUSTFLAGS="-C target-cpu=native"       compile-time    Production server (Sapphire Rapids+, Zen4+)
rustynum-avx2       RUSTFLAGS="-C target-feature=+avx2"    compile-time    Production laptop (Haswell+, 2013+)
rustynum-arm        cargo build --target aarch64-*          compile-time    Mac M1+, Graviton, RPi
```

### rustynum-universal (default, the auto binary)

This is what PR #102 builds. No flags. No target-cpu. No target-feature.

Every function in simd.rs uses `is_x86_feature_detected!` at runtime:
- AVX-512 detected → call simd_avx512 implementation
- AVX2 detected → call simd_avx2 implementation  
- Neither → call scalar fallback

`is_x86_feature_detected!` caches internally (one atomic read after first call).
Overhead: one predicted branch per function call. Negligible.

On non-x86 (ARM, WASM): the `#[cfg(target_arch = "x86_64")]` blocks
compile away. Only scalar path remains. Zero dead code in the binary.

This is the default. This is what `pip install rustynum` gets.
This is what CI tests. This is what works everywhere.

### rustynum-avx512 (production server)

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

LLVM sees `is_x86_feature_detected!("avx512f")` and knows the answer
is `true` at compile time. The check becomes a constant. The AVX2 and
scalar branches are dead-code eliminated. Zero dispatch overhead.
Same source code. Same simd.rs. Just smarter compilation.

### rustynum-avx2 (production laptop)

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

Same as above. LLVM eliminates AVX-512 and scalar branches.
Only AVX2 path remains.

### rustynum-arm (Apple Silicon, Graviton)

```bash
cargo build --release --target aarch64-unknown-linux-gnu
```

All `#[cfg(target_arch = "x86_64")]` blocks vanish.
Only scalar_fns remain. LLVM auto-vectorizes to NEON.

### Python wheels (maturin)

```bash
# Universal wheel (runtime dispatch):
maturin build --release

# Platform-specific wheels:
RUSTFLAGS="-C target-cpu=x86-64-v3" maturin build --release  # AVX2
RUSTFLAGS="-C target-cpu=x86-64-v4" maturin build --release  # AVX-512
maturin build --release --target aarch64-*                     # ARM
```

PyPI selects the right wheel per platform. The universal wheel
is the fallback for unknown platforms.

### The key insight

ONE source. ONE simd.rs with runtime dispatch. FOUR binaries.
The dedicated binaries are free — LLVM constant-folds the runtime
checks when the target features are known at compile time.
No code duplication. No separate code paths. No maintenance burden.

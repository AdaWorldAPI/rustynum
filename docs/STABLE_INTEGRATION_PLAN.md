# Stable Integration Plan — 4-Repository Cleanup

> **Created**: 2026-03-01
> **Author**: Claude session on `claude/implement-neural-network-research-owTKn`
> **Purpose**: Self-contained instructions for a future session to clean up
> ladybug-rs, crewai-rust, n8n-rs, and neo4j-rs for stable-only Rust with
> AVX-512 default + AVX2 silent fallback.

---

## Executive Summary

rustynum is **already on stable Rust** with a clean 3-tier SIMD dispatch:

```
AVX-512 VPOPCNTDQ → AVX2 Harley-Seal → scalar POPCNT
```

One binary, all CPUs. Runtime CPUID detection. Zero nightly dependencies.
Python `.so` inherits this — works on any x86-64 without recompile.

**The 4 downstream repos need to match:**
1. Delete duplicate SIMD implementations — call `rustynum_core::simd::*`
2. Fix zero-copy breaches — stop `.to_vec()`-ing Arrow buffers
3. Ensure toolchain = stable, CI = stable
4. Verify the AVX2 fallback path actually works (test on non-AVX-512 hardware)

---

## Architecture (ONE BINARY, ONE DISPATCH)

```
rustynum-core (types + SIMD + kernels)     ← STABLE, owns dispatch
    ↑
rustynum-holo (holographic containers)     ← STABLE
    ↑
ladybug-rs (BindSpace, storage, search)    ← NEEDS CLEANUP
    ↑
crewai-rust (Blackboard, agents)           ← NEEDS VERIFICATION
    ↑
n8n-rs (workflow orchestration)            ← NEEDS VERIFICATION
    ↑
neo4j-rs (graph persistence)              ← NEEDS VERIFICATION
```

**Law**: Dependencies flow ONE WAY (up). rustynum NEVER imports BindSpace.

---

## Pre-Conditions (ALREADY DONE — DO NOT REDO)

These are complete. Verify but do not re-implement:

| Item | Status | Evidence |
|------|--------|---------|
| `rust-toolchain.toml` = stable | DONE | `channel = "stable"` in rustynum root |
| Zero `#![feature(portable_simd)]` | DONE | grep confirms zero matches in any `.rs` file |
| `simd_compat.rs` wraps `std::arch` | DONE | Stable since Rust 1.89 |
| CI uses `dtolnay/rust-toolchain@stable` | DONE | `.github/workflows/rust.yml` |
| σ-significance scoring | DONE | `kernels.rs`: SignificanceLevel, SigmaScore, K2Histogram |
| CLAM → QualiaCAM integration | DONE | `rustynum-clam/src/qualia_cam.rs`, 49 tests |
| All rustynum tests pass | DONE | 1,253 tests, 0 failures (excluding qualia-xor) |

---

## rustynum Public SIMD API (What Other Repos Should Call)

### Runtime Dispatch (call ONCE, use many)

```rust
use rustynum_core::simd;

// One-time init — returns function pointer, caches CPUID result
let hamming_fn = simd::select_hamming_fn();   // → fn(&[u8], &[u8]) -> u64
let dot_i8_fn = simd::select_dot_i8_fn();     // → fn(&[u8], &[u8]) -> i64

// Hot path: zero branching, just fn pointer call
let distance = hamming_fn(query, candidate);
```

### Convenience Functions (inline dispatch per call)

```rust
// Single pair (acceptable for 1-off calls, not batch)
simd::hamming_distance(a: &[u8], b: &[u8]) -> u64
simd::popcount(a: &[u8]) -> u64
simd::dot_i8(a: &[u8], b: &[u8]) -> i64

// Batch (one CPUID check, N distance computations)
simd::hamming_batch(query, database, num_rows, row_bytes) -> Vec<u64>
simd::hamming_top_k(query, database, num_rows, row_bytes, k) -> (Vec<usize>, Vec<u64>)

// 3-tier cascade search with optional precision tail
simd::hdr_cascade_search(query, database, vec_bytes, num_vecs, threshold, PreciseMode) -> Vec<HdrResult>
```

### BLAS Level 1

```rust
simd::dot_f32(a: &[f32], b: &[f32]) -> f32
simd::dot_f64(a: &[f64], b: &[f64]) -> f64
simd::axpy_f32(alpha: f32, x: &[f32], y: &mut [f32])
simd::scal_f32(alpha: f32, x: &mut [f32])
simd::asum_f32(x: &[f32]) -> f32
simd::nrm2_f32(x: &[f32]) -> f32
```

### Capability Detection

```rust
use rustynum_core::compute;

let caps = compute::detect();  // Cached in OnceLock, checked once
caps.avx512f              // bool
caps.avx512_vpopcntdq     // bool
caps.avx512vnni           // bool
caps.avx512_bf16          // bool

let tier = compute::recommend_tier(m, n, k, Precision::Full);
// Returns: Int8Vnni | Bf16 | Fp32Avx512 | Scalar
```

---

## Repository 1: ladybug-rs

### Priority: HIGH (most SIMD duplication, most zero-copy breaches)

### Task 1a: Delete `ladybug-rs/src/core/simd.rs` (P0)

**What**: 348 lines duplicating rustynum's SIMD dispatch with compile-time
`#[cfg(target_feature)]` gates instead of runtime detection.

**Contains**: `hamming_distance()`, `hamming_avx512()`, `hamming_avx2()`,
`HammingEngine` — all reimplementing what `rustynum_core::simd` already does.

**Steps**:
1. `grep -rn "mod simd" ladybug-rs/src/` — find the module declaration
2. `grep -rn "use.*core::simd" ladybug-rs/src/` — find all call sites
3. Replace every `crate::core::simd::hamming_distance(a, b)` with
   `rustynum_core::simd::hamming_distance(a, b)`
4. Replace `HammingEngine` usage with `rustynum_core::simd::select_hamming_fn()`
5. Delete `ladybug-rs/src/core/simd.rs`
6. Remove `mod simd;` from `ladybug-rs/src/core/mod.rs`

**Verification**: `cargo test -p ladybug-rs` — all search/distance tests must pass

### Task 1b: Fix scalar Hamming loop in bind_space.rs (P0)

**Location**: `ladybug-rs/src/storage/bind_space.rs:1848-1855`

**What**: Manual scalar XOR+popcount loop instead of calling rustynum.

**Fix**: Replace the loop with:
```rust
let distance = rustynum_core::simd::hamming_distance(a_bytes, b_bytes);
```

### Task 1c: Fix `.to_vec()` in rustynum_accel.rs (P0)

**Location**: `ladybug-rs/src/core/rustynum_accel.rs:148`

**What**: `container_bundle()` copies data with `.to_vec()`.

**Fix**: Accept `&[u8]` reference instead of copying to owned Vec.

### Task 1d: Verify toolchain (P1)

1. Check `rust-toolchain.toml` — should be `channel = "stable"` (may already be)
2. Check CI workflow — should use `dtolnay/rust-toolchain@stable`
3. `grep -rn 'feature.*portable_simd\|feature.*simd' ladybug-rs/src/` — must be zero
4. `grep -rn 'cfg.*target_feature' ladybug-rs/src/` — should only be in test code (not hot paths)

### Task 1e: Wire rustynum_accel into core search (P1)

**Location**: `ladybug-rs/src/core/rustynum_accel.rs`

**What**: `fingerprint_hamming()`, `container_hamming()`, `slice_hamming()` exist
but are only called from `python/mod.rs:39`. They should be the primary search path.

**Fix**: In `ladybug-rs/src/storage/bind_space.rs` search functions, replace
inline distance computation with calls through `rustynum_accel`.

---

## Repository 2: crewai-rust

### Priority: MEDIUM (uses rustynum indirectly via ladybug-rs)

### Task 2a: Verify no nightly features (P1)

```bash
grep -rn '#!\[feature' crewai-rust/src/
grep -rn 'portable_simd' crewai-rust/src/
cat crewai-rust/rust-toolchain.toml
```

Expected: zero matches. If any exist, remove them.

### Task 2b: Verify CI uses stable (P1)

Check `.github/workflows/*.yml` — build/test should use `dtolnay/rust-toolchain@stable`.
Miri can use nightly (standard practice).

### Task 2c: Check SIMD usage (P1)

crewai-rust should NOT have its own SIMD code. It reads/writes through
`TypedSlots` on the Blackboard. All compute goes through rustynum.

```bash
grep -rn 'std::arch\|is_x86_feature_detected\|target_feature' crewai-rust/src/
```

Expected: zero matches. If any exist, replace with rustynum dispatch calls.

### Task 2d: Check jit_link.rs → jitson connection (P2)

`crewai-rust/src/persona/jit_link.rs` produces `JitScanParams` for jitson.
Verify it doesn't depend on nightly features and that the interface is
compatible with rustynum's stable types.

---

## Repository 3: n8n-rs

### Priority: MEDIUM (workflow layer, should have zero SIMD)

### Task 3a: Verify no nightly features (P1)

```bash
grep -rn '#!\[feature' n8n-rs/src/
grep -rn 'portable_simd' n8n-rs/src/
cat n8n-rs/rust-toolchain.toml
```

Expected: zero matches. n8n-rs is an orchestration layer — it should never
touch SIMD directly.

### Task 3b: Verify CI uses stable (P1)

Check `.github/workflows/*.yml` — should use stable for build/test.

### Task 3c: Check Arrow zero-copy chain (P1)

n8n-rs has an "Arrow Zero-Copy Chain" (referenced in CLAUDE.md § 12).
Verify it uses `CogRecordView<'a>` (borrowing) not `CogRecord` (copying).

```bash
grep -rn '\.to_vec()\|\.clone()' n8n-rs/src/ | grep -i 'arrow\|record\|buffer'
```

Any hits in hot paths need fixing.

---

## Repository 4: neo4j-rs

### Priority: LOW (persistence layer, pure graph ops)

### Task 4a: Verify no nightly features (P1)

```bash
grep -rn '#!\[feature' neo4j-rs/src/
grep -rn 'portable_simd' neo4j-rs/src/
cat neo4j-rs/rust-toolchain.toml
```

Expected: zero matches. neo4j-rs is a Cypher/Bolt client — no SIMD.

### Task 4b: Verify CI uses stable (P1)

Check `.github/workflows/*.yml` — should use stable for build/test.

### Task 4c: Check for `.to_vec()` on fingerprint data (P2)

If neo4j-rs stores/retrieves fingerprints, verify it doesn't copy them
unnecessarily during serialization.

```bash
grep -rn '\.to_vec()\|\.clone()' neo4j-rs/src/ | grep -i 'fingerprint\|container\|overlay'
```

---

## Zero-Copy Breaches IN rustynum (for reference)

These exist within rustynum itself. Fix them before or alongside the cross-repo work:

### P0 — Hot Path Copies

| File | Line(s) | Pattern | Cost |
|------|---------|---------|------|
| `rustynum-arrow/src/arrow_bridge.rs` | 152-155 | 4× `.to_vec()` in `record_batch_to_cogrecords()` | 819 MB/100K records |
| `rustynum-arrow/src/indexed_cascade.rs` | 94-97 | 4× `extend_from_slice()` in `build()` | 819 MB/100K records |
| `rustynum-core/src/spatial_resonance.rs` | 157-159 | 3× `.to_vec()` in `from_flat_bytes()` | 6 KB/crystal |

**Fix for arrow_bridge.rs**: `cogrecord_views()` (line 205-235) already exists
and returns `Vec<CogRecordView<'a>>` with borrowed `&'a [u8]` fields. Change
callers from `record_batch_to_cogrecords()` → `cogrecord_views()`.

**Fix for indexed_cascade.rs**: `build_from_arrow()` (line 119-161) already
exists and uses `column_flat_data()` for zero-copy. Change callers from
`build()` → `build_from_arrow()`.

**Fix for spatial_resonance.rs**: Change `CrystalAxis::from_bf16_bytes()`
signature from `Vec<u8>` to `&[u8]`.

### P1 — Moderate Copies

| File | Pattern | Notes |
|------|---------|-------|
| `rustynum-rs/src/num_array/cogrecord.rs:280-283` | `from_borrowed()` calls `.to_vec()` despite name | Misleading API name |
| `rustynum-holo/src/cogrecord_v3.rs:82-85` | Constructor copies meta/btree/embed | Accept `&[u8]` instead |
| `rustynum-rs/src/num_array/linalg.rs` | `.to_vec()` in matrix operations | Mathematical library code |
| `rustynum-rs/src/num_array/statistics.rs` | `.to_vec()` in statistical functions | Not time-critical |

---

## Verification Checklist (Run After All Changes)

### Per-Repository

```bash
# 1. No nightly features
grep -rn '#!\[feature' src/ | grep -v test  # Must be empty
cat rust-toolchain.toml                      # Must say "stable"

# 2. No duplicate SIMD
grep -rn 'is_x86_feature_detected\|cfg.*target_feature' src/ | grep -v test  # Must be empty

# 3. Tests pass
cargo test

# 4. Clippy clean
cargo clippy --workspace -- -D warnings

# 5. No unnecessary copies in hot paths
grep -rn '\.to_vec()' src/ | grep -v test | grep -v 'serializ\|wire\|owned'
```

### Cross-Repository Integration Test

After all 4 repos are cleaned up, test the full stack:

```bash
# Build rustynum (the substrate)
cd rustynum && cargo build --release

# Build ladybug-rs against rustynum
cd ladybug-rs && cargo build --release

# Build crewai-rust against ladybug-rs
cd crewai-rust && cargo build --release

# Verify Python bindings work on AVX2-only machine
cd rustynum/bindings/python && cargo build --release
python -c "import rustynum; print(rustynum.hamming_distance(b'\x00'*2048, b'\xff'*2048))"
```

### AVX2 Fallback Verification

To verify the AVX2 fallback works correctly:

```bash
# Force AVX2-only mode by unsetting AVX-512 (if hardware supports it):
# Option A: Run on a machine without AVX-512 (e.g., Ryzen, older Xeon)
# Option B: Use Intel SDE to emulate AVX2-only:
#   sde64 -skl -- cargo test -p rustynum-core -- hamming
#
# What to check:
# 1. hamming_distance() returns correct values
# 2. hamming_batch() matches scalar reference
# 3. hdr_cascade_search() produces same results as AVX-512 path
# 4. Python bindings work without AVX-512
```

---

## Execution Order

1. **ladybug-rs** first (most work, most impact)
   - Task 1a: Delete simd.rs duplicate
   - Task 1b: Fix scalar loop in bind_space.rs
   - Task 1c: Fix .to_vec() in rustynum_accel.rs
   - Task 1d: Verify toolchain
   - Task 1e: Wire rustynum_accel into search
   - Run: `cargo test -p ladybug-rs`

2. **rustynum zero-copy fixes** (can be parallel with ladybug-rs)
   - Switch callers to `cogrecord_views()` / `build_from_arrow()`
   - Fix `CrystalAxis::from_bf16_bytes()` signature
   - Run: `cargo test --workspace --exclude qualia-xor`

3. **crewai-rust** (fast — mostly verification)
   - Tasks 2a-2d: verify stable, no SIMD, clean CI
   - Run: `cargo test`

4. **n8n-rs** (fast — mostly verification)
   - Tasks 3a-3c: verify stable, no SIMD, Arrow zero-copy
   - Run: `cargo test`

5. **neo4j-rs** (fast — mostly verification)
   - Tasks 4a-4c: verify stable, no copies
   - Run: `cargo test`

6. **Cross-stack integration test**
   - Build all 5 repos in dependency order
   - Verify Python bindings on AVX2-only

---

## What NOT to Change

- **Archive crates** (rustynum-archive, rustynum-archive-v3, rustynum-carrier, rustynum-focus) — frozen, sacred
- **jitson/** — separate workspace, NOT a member, builds independently
- **qualia-xor** — excluded from CI, separate candle/tokenizers dependency issue
- **rustynum-core SIMD dispatch** — already correct, do not restructure
- **K0/K1/K2 pipeline** — kernels.rs is production-ready
- **σ-significance scoring** — SignificanceLevel/SigmaScore fully implemented
- **CLAM/QualiaCAM** — 49 tests, fully wired

---

## Notes for the Session

- rustynum owns the SIMD dispatch. Other repos MUST NOT reimplements it.
- The dispatch pattern is: detect ONCE → function pointer → hot loop. No branching.
- `is_x86_feature_detected!()` in a hot loop = WRONG. Use `select_hamming_fn()`.
- `#[cfg(target_feature = "avx512f")]` = compile-time gate = WRONG for library code.
  It means the binary won't work on machines without AVX-512. Use runtime detection.
- `.to_vec()` on Arrow buffers = WRONG. Use `column_flat_data()` or `CogRecordView<'a>`.
- The Python bindings already get AVX2 fallback for free — the `.so` dispatches at runtime.

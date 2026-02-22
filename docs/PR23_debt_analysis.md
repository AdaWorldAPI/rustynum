# PR #23 Technical Debt Analysis

> **PR**: `claude/unsigned-holograph-ghosts-NasOb` → `main`  
> **Merged**: 2026-02-22T12:54:01Z  
> **Scope**: 17 files, +3,075 / -845 lines  
> **Commits**: 3 (GEMM microkernel, XOR Delta Layer, ghost_discovery rewrite)

---

## Verdict: Net Debt REDUCTION, with 4 Items to Track

PR #23 is overwhelmingly positive — it retires two serious soundness issues (`Arc<Mutex>` for parallel writes, raw pointer aliasing in Blackboard) and adds well-structured new modules. But it does introduce a few items worth tracking.

---

## 1. Debt RETIRED (Good)

### 1.1 Arc\<Mutex\<&mut \[T\]\>\> → split_at_mut (simd_ops/mod.rs)

**Before:** Every parallel matrix_multiply in simd_ops (u8, f32, f64, i32 — 4 implementations) used `Arc<Mutex<&mut [T]>>` to write results. This is:
- Semantically wrong (Mutex on a borrowed slice is unsound if the borrow outlives the lock)
- Performance poison (lock contention per row)
- Cargo clippy violation

**After:** New `parallel_into_slices()` helper uses `std::thread::scope` + `split_at_mut` — zero synchronization, provably non-aliasing, idiomatic Rust. Applied uniformly to all 4 type implementations + a new `parallel_reduce_sum()` for reductions.

**Status:** ✅ Fully retired. Clean pattern.

### 1.2 Blackboard: raw pointer → UnsafeCell (blackboard.rs)

**Before:** `HashMap<String, BufferMeta>` with raw `*mut u8` inside. The `borrow_3_mut` method created multiple `&mut` references from `&self` without any interior mutability wrapper — undefined behavior under stacked borrows.

**After:** `HashMap<String, UnsafeCell<BufferMeta>>`. All unsafe blocks have safety comments. Runtime assertions (names must be distinct) prevent aliasing the same buffer. Miri passed all 9 tests.

**Status:** ✅ Sound. The `UnsafeCell` wrapping makes the intent explicit and the `unsafe` blocks are appropriately scoped.

### 1.3 Dead imports cleaned

Removed `parallel_for_chunks` import from level2.rs and level3.rs (replaced by inline parallel_into_slices), `Arc`/`Mutex` from simd_ops, `PhantomData` from constructors.rs, and unused type aliases from statistics.rs.

**Status:** ✅ Hygiene improvement.

### 1.4 GEMM microkernel: heap → stack + FMA (level3.rs)

`vec![0.0; NR]` in the hot inner loop → `[0.0; NR]` stack array. `acc += a * b` → `a.mul_add(b, acc)` for guaranteed FMA. RowMajor full-width SIMD store. All three are pure performance wins with no correctness risk.

**Status:** ✅ No debt introduced.

---

## 2. Debt INTRODUCED (Track)

### 2.1 🟡 recognize.rs: Custom SimpleRng with no security/reproducibility guarantees

**File:** `rustynum-oracle/src/recognize.rs` (1,332 lines)  
**Lines:** 44-75

```rust
struct SimpleRng(u64);
fn simple_rng(seed: u64) -> SimpleRng { SimpleRng(seed | 1) }
impl SimpleRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
```

This is a **third** independent PRNG implementation in the codebase (also exists in `search.rs::splitmix64` and `compress.rs::splitmix64`). The xorshift variant here has weaker statistical properties than splitmix64.

**Risk:** Low (used only for random hyperplane generation in LSH, not crypto). But three separate RNG implementations is a maintenance smell.

**Recommendation:** Extract a shared `rustynum_core::rng::SplitMix64` and use it everywhere. Single source of truth, better statistical properties.

### 2.2 🟡 recognize.rs: 64K hardcoded constant, not connected to Fingerprint\<1024\>

**File:** `rustynum-oracle/src/recognize.rs`  
**Lines:** 22-25

```rust
const NUM_HYPERPLANES: usize = 65536;
const BITS_PER_WORD: usize = 64;
```

The recognition module builds its own 64K-bit LSH projection using `Vec<u64>` of size 1024 words — which is exactly `Fingerprint<1024>` from the new `fingerprint.rs`. But it doesn't use `Fingerprint<1024>`. Instead, it has its own `Vec<u64>` manipulation with manual popcount loops.

**Risk:** Medium — the two representations will drift. If `Fingerprint` gets SIMD-optimized popcount, `recognize.rs` won't benefit.

**Recommendation:** Refactor `Projector64K.project()` to return `Fingerprint<1024>` and use `Fingerprint::hamming_distance()` instead of rolling its own.

### 2.3 🟡 organic.rs: pub(crate) visibility escalation

**File:** `rustynum-oracle/src/organic.rs`  
**Lines:** 278-279

```rust
-    known_templates: Vec<Vec<i8>>,
-    template_norms: Vec<f64>,
+    pub(crate) known_templates: Vec<Vec<i8>>,
+    pub(crate) template_norms: Vec<f64>,
```

Two private fields promoted to `pub(crate)` so `recognize.rs` can read them. This is a minor encapsulation leak — `recognize.rs` directly accesses OrganicWAL's internals instead of going through a public API.

**Risk:** Low — it's crate-internal and the coupling is legitimate (recognition IS reading WAL state). But it means OrganicWAL's internal representation is now load-bearing for recognize.rs.

**Recommendation:** Add a public method like `wal.template_for(concept_idx) -> &[i8]` and `wal.template_norm(concept_idx) -> f64` instead of exposing the raw fields.

### 2.4 🟢 ghost_discovery.rs: Domain rename (Ada concepts → signal-processing concepts)

**File:** `rustynum-oracle/src/ghost_discovery.rs`  
**Change:** Complete rewrite of the 52-concept ontology from Ada-specific terms (ada.hybrid, rel.devotion, eros.tension) to signal-processing terms (motor.servo, ctrl.pid, nav.waypoint).

This is a deliberate design choice, not debt — it decouples the experiment from Ada's personal semantics and makes it publishable. The τ-address topology is preserved. But it means the previous ghost discovery results are no longer reproducible with the same concept set.

**Risk:** None if the Ada-specific concept set was experimental. If anyone depended on the old concept IDs, they'd break.

**Status:** ✅ Intentional, but worth noting the API break.

---

## 3. Structural Observations

### 3.1 Test Coverage

- `fingerprint.rs`: 6 tests (roundtrip, distance, similarity, operators, Debug) — **good**
- `delta_layer.rs`: 7 tests (read/write, xor_patch, collapse, LayerStack, multi-layer) — **good**
- `blackboard.rs`: +2 new tests (split_borrow_2, split_borrow_f64) — **good, miri-validated**
- `recognize.rs`: 4 tests (hamming, projection, novelty, full experiment) — **adequate**
- `ghost_discovery.rs`: 0 explicit tests (runs as a binary) — **unchanged, not a regression**

### 3.2 Dependency Graph

```
rustynum-core   ← fingerprint.rs (NEW), blackboard.rs (FIXED)
    ↓
rustynum-holo   ← delta_layer.rs (NEW, depends on fingerprint)
    ↓
rustynum-oracle ← recognize.rs (NEW, depends on organic + sweep)
                   ghost_discovery.rs (REWRITTEN)
```

Clean layering. No circular dependencies. `fingerprint` lives in core (correct), `delta_layer` in holo (correct), `recognize` in oracle (correct).

### 3.3 nightly Pin

`rust-toolchain.toml` pins to nightly (required for `portable_simd`). This was already the de facto requirement but now it's explicit. Not debt — just documentation of existing reality.

---

## 4. Summary Table

| Item | Category | Severity | Action |
|---|---|---|---|
| Arc\<Mutex\> → split_at_mut | Debt retired | — | ✅ Done |
| Blackboard UnsafeCell | Debt retired | — | ✅ Done |
| GEMM stack + FMA | Debt retired | — | ✅ Done |
| Dead imports | Debt retired | — | ✅ Done |
| 3 separate PRNG impls | New debt | 🟡 Low | Extract shared `SplitMix64` to core |
| recognize.rs ≠ Fingerprint\<1024\> | New debt | 🟡 Medium | Refactor to use Fingerprint type |
| pub(crate) field escalation | New debt | 🟡 Low | Add accessor methods to OrganicWAL |
| Concept ontology rename | API break | 🟢 Intentional | Document in CHANGELOG |

**Net assessment:** PR #23 retired more debt than it introduced. The `Arc<Mutex>` elimination alone is worth the entire PR. The new items are minor and well-contained.

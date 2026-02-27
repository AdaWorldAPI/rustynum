# CLAUDE.md — rustynum

> **Last Updated**: 2026-02-25
> **Branch**: `claude/vsaclip-hamming-recognition-y0b94`
> **Owner**: Jan Hübener (jahube)

---

## READ THIS FIRST — Role in the Four-Level Architecture

rustynum is **Level 1 — Surface** (spatial substrate).

> **Canonical cross-repo architecture:** [ada-docs/architecture/FOUR_LEVEL_ARCHITECTURE.md](https://github.com/AdaWorldAPI/ada-docs/blob/main/architecture/FOUR_LEVEL_ARCHITECTURE.md)

rustynum is the **hardware layer AND the bindspace type owner**. It provides:

1. **SIMD-dispatched kernels** for distance computation (Hamming, BF16, dot)
2. **The unified bindspace surface types**: `Fingerprint<256>`, `DeltaLayer`,
   `LayerStack`, `CollapseGate`, `AlignedBuf2K`, `MultiOverlay`
3. **Holographic containers**: `Overlay`, Gabor wavelets, spectral analysis

All crates compile into **ONE binary**. The bindspace surface is the SAME
memory — never copied between crates. `Overlay.buffer` (2048 bytes) IS
`Fingerprint<256>` (256 × u64 = 2048 bytes) viewed through `as_fingerprint_words()`.
No conversion. No copy. Same pointer.

**Dependency direction (LAW — do not violate):**

```
rustynum-core (types + SIMD)
    ↑
rustynum-holo (holographic containers, uses Fingerprint<256> directly)
    ↑
ladybug-rs (BindSpace, CollapseGate decisions, storage)
    ↑
crewai-rust (Blackboard, agents — reads/writes via TypedSlots)
```

rustynum NEVER imports BindSpace, crewai-rust, n8n-rs, or neo4j-rs.
It is a compute + type leaf with zero IO. Arrow points ONE way.

---

## 1. Workspace Structure

```
rustynum/
├── rustynum-core/       # SIMD, BF16, kernels, Fingerprint, DeltaLayer, LayerStack, CollapseGate
├── rustynum-holo/       # Holographic containers: Overlay, MultiOverlay, AlignedBuf2K, Gabor
├── rustynum-rs/         # CogRecord, Python bindings bridge, ndarray ops
├── rustynum-arrow/      # Arrow bridge, indexed cascade, horizontal sweep
├── rustyblas/           # BLAS: GEMM (f32, bf16, int8), level1-3
├── rustymkl/            # MKL-like interface (optional)
├── rustynum-oracle/     # Sweetspot evaluation for 3D vector sizes
├── rustynum-clam/       # CLAM integration
├── qualia_xor/          # Qualia corpus experiments (Nib4/BERT, Cypher VSA, hydrate-agents)
├── bindings/python/     # PyO3 bindings
├── jitson/              # Cranelift JIT (own workspace — NOT a member)
│
│   # Archive crates — INTENTIONAL frozen snapshots, DO NOT DELETE
├── rustynum-archive/    # V1 reference implementation
├── rustynum-archive-v3/ # V3 reference implementation
├── rustynum-carrier/    # Carrier wave experiments
└── rustynum-focus/      # Focus/attention experiments
```

### Workspace Isolation

`jitson/` has its own `[workspace]` in its Cargo.toml. It is NOT a
workspace member. Do NOT add it to the root `[workspace].members`.
Do NOT add `exclude = ["jitson"]` — that causes Cargo to read its
deps which pull in the entire wasmtime workspace.

### Archive Crates Are Sacred

`rustynum-archive`, `rustynum-archive-v3`, `rustynum-carrier`, `rustynum-focus`
are **intentional frozen reference implementations** for scientific
reproducibility. They are NOT copy-paste debt. Their `path = "../rustynum-core"`
deps ensure they compile against current infrastructure.

**DO NOT delete them. DO NOT "clean them up". DO NOT refactor them.**

---

## 2. The SIMD Dispatch Pattern (CANONICAL)

```rust
// One-time CPUID detection at init (cached in static)
let hamming_fn = select_hamming_fn();   // AVX-512 VPOPCNTDQ / AVX2 Harley-Seal / scalar
let dot_fn = select_dot_i8_fn();        // AVX-512 VNNI VPDPBUSD / scalar
let bf16_fn = select_bf16_hamming_fn(); // AVX-512 BITALG / scalar

// Hot path: function pointer call, zero branching
let distance = hamming_fn(a_bytes, b_bytes);
```

**Rule: Detection happens ONCE. The hot path is a function pointer call.**
No `if is_x86_feature_detected!()` in the hot loop.

### Tiered Dispatch

```
Tier 0: INT8 Prefilter  (VNNI vpdpbusd)      — 90% pruned, cheapest
Tier 1: Binary Hamming   (VPOPCNTDQ)          — HDC distance, 2ns/2KB
Tier 2: BF16 Structured  (BITALG vpshufb)     — sign/exp/man weighted
Tier 3: FP32 AVX-512     (vmulps/vaddps)       — full precision, expensive
```

---

## 3. The 3-Kernel Pipeline (kernels.rs)

LIBXSMM-inspired fixed-size cascade for bitpacked containers:

```
K0 Probe (64-bit):   XOR + POPCNT on 1 u64    → eliminates ~55%
K1 Stats (512-bit):  XOR + POPCNT on 8 u64    → eliminates ~90% of survivors
K2 Exact (full):     XOR + AND + POPCNT        → EnergyConflict decomposition
```

Two fixed SKUs (no dynamic sizing):
- **SKU-16K**: 16384 bits = 256 words = 2048 bytes (CogRecord standard)
- **SKU-64K**: 65536 bits = 1024 words = 8192 bytes (CogRecord8K full)

K2 returns `EnergyConflict` — not just Hamming distance:
- `conflict`: bits where a=1,b=1 (agreement) vs a=0,b=0 (absence)
- `energy_a`, `energy_b`: popcount of each input
- Kills negative cancellation (a key insight)

---

## 4. The Hybrid Pipeline (hybrid.rs)

Bridges kernels + BF16 + awareness into one pipeline:

```
Tier 0 Prefilter (optional)
  → K0 → K1 → K2 (integer Hamming, ~95% pruned)
    → BF16 Tail (survivors only, ~5%)
         ├─ Structured distance: sign/exp/man weighted
         ├─ Structural diff: which dimensions changed
         └─ Awareness: crystallized/tensioned/uncertain/noise
              └→ Learning feedback: attention weights → WideMetaView W144-W159
```

### Awareness Substrate (bf16_hamming.rs)

Per-dimension classification from BF16 decomposition:

| State | Bits | Meaning |
|-------|------|---------|
| Crystallized | 00 | Sign + exp agree → settled knowledge |
| Tensioned | 01 | Sign disagrees → active contradiction |
| Uncertain | 10 | Sign agrees, high exp spread → direction known |
| Noise | 11 | Only mantissa differs → irrelevant |

These 2-bit states pack into `Vec<u8>` (4 dimensions per byte).

---

## 5. Spatial Resonance (spatial_resonance.rs)

3D BF16 crystal for SPO (Subject-Predicate-Object) encoding:

```
SpatialCrystal3D
├── X-axis: CrystalAxis (2048 bytes) = Subject
├── Y-axis: CrystalAxis (2048 bytes) = Predicate
└── Z-axis: CrystalAxis (2048 bytes) = Object
```

SPO encoding via XOR bind:
```
X = S ⊕ P   (Subject bound with Predicate)
Y = P ⊕ O   (Predicate bound with Object)
Z = S ⊕ O   (Subject bound with Object)
```

Recovery (XOR is self-inverse):
```
S = X ⊕ P   (unbind Predicate from X)
O = Y ⊕ P   (unbind Predicate from Y)
```

### Sweet Spot Evaluation (rustynum-oracle)

The `sweetspot` binary evaluates 3D vector sizes:
- **D = 8192 bits** (1024 BF16 dims = 2048 bytes) per axis is optimal
- **Base = Signed(5)** quantization
- **K = 3–13 concepts** for < 0.1 error

This aligns with Jina 1024-D embeddings, ladybug-rs Crystal4K projections,
and VSACLIP SKU-16K container standard.

---

## 6. Public API Contract

All rustynum compute functions take `&[u8]` or `&[u64]` slices. They NEVER
allocate, NEVER do IO, NEVER access the network. Pure compute.

### Compute Functions

| Function | Input | Output | File |
|----------|-------|--------|------|
| `select_hamming_fn()` | — | `fn(&[u8],&[u8])->u32` | simd.rs |
| `select_dot_i8_fn()` | — | `fn(&[i8],&[i8])->i32` | simd.rs |
| `select_bf16_hamming_fn()` | — | `fn(&[u8],&[u8])->BF16Result` | bf16_hamming.rs |
| `k0_probe()` | `&[u64],&[u64]` | `u32` | kernels.rs |
| `k1_stats()` | `&[u64],&[u64]` | `u32` | kernels.rs |
| `k2_exact()` | `&[u64],&[u64]` | `EnergyConflict` | kernels.rs |
| `hybrid_pipeline()` | config + candidates | `Vec<HybridScore>` | hybrid.rs |
| `SpatialCrystal3D::spo_encode()` | S,P,O | Crystal | spatial_resonance.rs |

### Unified Bindspace Surface Types (rustynum-core)

These types are the SAME surface across all crates. One binary, one memory.

| Type | Size | Purpose | File |
|------|------|---------|------|
| `Fingerprint<256>` | 2048 bytes | The universal container = `[u64; 256]` | fingerprint.rs |
| `DeltaLayer<N>` | Fingerprint + writer_id | XOR delta from ground truth — writer owns `&mut` | delta.rs |
| `LayerStack<N>` | ground + Vec<DeltaLayer> | Multi-writer concurrent state | layer_stack.rs |
| `CollapseGate` | enum | Flow/Hold/Block decision | layer_stack.rs |

### Holographic Surface Types (rustynum-holo)

| Type | Size | Purpose | File |
|------|------|---------|------|
| `Overlay` | 2048 bytes | IS `Fingerprint<256>` via `as_fingerprint_words()` | holograph.rs |
| `AlignedBuf2K` | 2048 bytes | `repr(align(8))` buffer with guaranteed zero-copy view | holograph.rs |
| `MultiOverlay` | N × Overlay | One per agent, conflict via AND+popcount | holograph.rs |

### The Zero-Copy Rule

`Overlay.as_fingerprint_words()` returns `&[u64; 256]` — a pointer reinterpret
of the same 2048 bytes. **Never copy between these types in-process.**
`to_bytes()` / `from_bytes()` exist ONLY for wire serialization (disk, network).
Within the binary, everything is a view.

---

## 7. Technical Debt — Prioritized

### P0 — Must Fix (Blocks Production)

| Debt | Location | Impact | Fix |
|------|----------|--------|-----|
| **34 public API panics** | `rustynum-rs/src/operations.rs` | Panic in library code | Replace `assert!`/`panic!` with `Result<_, NumError>` |
| **GEMM bounds check** | `rustyblas/src/int8_gemm.rs` | Unchecked buffer access | Add `assert!(c.len() == m * n)` at entry |

### P1 — Should Fix (Quality)

| Debt | Location | Impact | Fix |
|------|----------|--------|-----|
| Zero-copy violations | `rustynum-arrow/src/arrow_bridge.rs` | Unnecessary allocations | Use Arrow buffer references |
| Python binding f32/f64 copy-paste | `bindings/python/` | Duplicated code | Generic impl or macro |

### P2 — Nice to Have (Parity)

| Debt | Location | Impact | Fix |
|------|----------|--------|-----|
| Broadcasting | `operations.rs` | ndarray parity gap | Implement shape broadcast |
| `s![]` macro | — | Ergonomic slicing | Implement index macro |
| Compile-time dimensions | — | Shape safety | Add typenum dims |

### Confirmed Sound

| Area | Validation | Status |
|------|------------|--------|
| SIMD core | Miri pass (PR #59) | Sound |
| unsafe discipline | `// SAFETY:` on all blocks | Documented |
| GEMM routing | Fixed (diagonal regression) | Correct |
| Kernel pipeline (K0/K1/K2) | 17 tests | Passing |
| Hybrid pipeline | 8 tests | Passing |
| BF16 awareness | 8 tests | Passing |
| Blackboard `&self→&mut` UB | Fixed: uses `&mut self` | Sound |
| u8 matmul `Arc<Mutex>` | Fixed: `parallel_into_slices` + `split_at_mut` | Lock-free |
| Trait bound copy-paste | Fixed: `NumElement` supertrait | Clean |
| XOR Delta Layer | Implemented: Fingerprint + DeltaLayer + LayerStack + CollapseGate | 47 tests |
| Holographic zero-copy | `Overlay.as_fingerprint_words()` = pointer reinterpret | No copy |

### Structural Race Prevention

Race conditions cannot exist by construction:
- **Ground truth is `&self` forever** during processing cycles
- **Each writer owns their own delta as `&mut`** — standard Rust ownership
- **`split_at_mut`** for contiguous byte regions (GEMM rows, Z-slabs)
- **XOR Delta Layers** for scattered binary vector mutations (fingerprints)
- If a race condition appears, the architecture is wrong — fix the design, not the symptom

### CollapseGate = Airlock (Luftschleuse)

Deltas ARE superposition — they coexist over ground truth without collapsing.
The CollapseGate is the airlock between superposition and ground truth.

**Two kinds of XOR — never confuse them:**

| XOR | When | Target | Borrow |
|-----|------|--------|--------|
| **Delta XOR** | WRITE phase | Writer's own `DeltaLayer` | `&mut DeltaLayer` (private) |
| **Commit XOR** | After gate FLOW | Ground truth | `&mut LayerStack` (exclusive) |

**Phase ordering (strict):**

```
1. WRITE         Each writer delta-XORs their intent into their own layer.
                 Ground is &self — untouched. Each delta is &mut — private.
                      │
2. AWARENESS     Read superposition: ground ^ delta[0] ^ delta[1] ^ ...
                 AND + popcount SEES contradictions between deltas.
                 Contradictions ARE the awareness signal.
                 Without contradiction there is nothing to be aware OF.
                 Everything is &self — nobody writing.
                      │
3. GATE          CollapseGate evaluates awareness → decision.
                      │
            ┌─────────┼─────────┐
          FLOW       HOLD      BLOCK
            │         │         │
4. COMMIT   │    keep super-  discard
   ground   │    position     super-
   ^= Σδ   │    (accumulate  position
            │    more evidence)
       collapse
       to ground
       truth
```

- **WRITE → AWARENESS**: you need the contradiction to have the awareness.
  Awareness is reading the superposition. Without superposition, no signal.
- **AWARENESS → GATE**: gate uses awareness (AND+popcount) to decide.
- **GATE → COMMIT**: XOR to ground ONLY on FLOW. This is the only `&mut` on ground.
- **HOLD**: superposition persists. Next cycle adds more deltas. Awareness grows.
- **BLOCK**: superposition discarded. Ground unchanged. Start fresh.

XOR is the algebra. Bundle is the superposition. Awareness reads the bundle.
The gate is the decision. Commit is the only write to ground.

---

## 8. Enforced Practices

### Unsafe Discipline

Every `unsafe` block MUST have a `// SAFETY:` comment explaining:
1. Why unsafe is needed (what safe Rust can't express)
2. What invariants are maintained
3. What could go wrong if invariants are violated

```rust
// SAFETY: CPUID check above guarantees AVX-512 VPOPCNTDQ is available.
// Input slices are the same length (asserted at function entry).
// Output is a u32 popcount — no memory safety concern.
unsafe { _mm512_popcnt_epi64(xor_result) }
```

### Test Discipline

- Hardware-conditional assertions: do NOT hardcode `assert!(caps.avx512f)`
- Use `if caps.avx512f { assert!(...) }` pattern
- CI runners may lack AVX-512 — tests must pass on scalar fallback

### Version Discipline

| Item | Value |
|------|-------|
| `rust-version` | 1.93 (all crates) |
| `edition` | 2021 (core), 2024 acceptable for leaf |
| Arrow | 57 |
| DataFusion | 51 |
| Nightly | NEVER — nightly changes deps across the whole stack |

---

## 9. Key Files

### Bindspace Surface (shared by all crates in one binary)

| File | Purpose |
|------|---------|
| `rustynum-core/src/fingerprint.rs` | `Fingerprint<N>` — THE container type, `[u64; N]` |
| `rustynum-core/src/delta.rs` | `DeltaLayer<N>` — XOR delta, writer owns `&mut`, ground is `&self` |
| `rustynum-core/src/layer_stack.rs` | `LayerStack<N>` + `CollapseGate` — multi-writer + transparent writethrough with bundle |
| `rustynum-holo/src/holograph.rs` | `Overlay` (IS `Fingerprint<256>` via `as_fingerprint_words()`), `MultiOverlay`, `AlignedBuf2K` |

### SIMD Compute

| File | Lines | Purpose |
|------|-------|---------|
| `rustynum-core/src/simd.rs` | ~900 | VPOPCNTDQ, VNNI, Harley-Seal, dispatch |
| `rustynum-core/src/bf16_hamming.rs` | ~500 | BF16 structured distance + awareness |
| `rustynum-core/src/kernels.rs` | ~1000 | K0/K1/K2 pipeline, EnergyConflict |
| `rustynum-core/src/hybrid.rs` | ~470 | Hybrid pipeline + Tier 0 prefilter |
| `rustynum-core/src/spatial_resonance.rs` | ~700 | 3D BF16 crystal + SPO encoding |
| `rustynum-core/src/compute.rs` | ~200 | CPUID detection, tier recommendation |
| `rustynum-arrow/src/horizontal_sweep.rs` | ~770 | 90° word-by-word early exit |
| `rustynum-arrow/src/indexed_cascade.rs` | ~400 | Indexed Hamming cascade |
| `rustynum-rs/src/operations.rs` | ~800 | ndarray ops (P0: needs Result types) |
| `rustyblas/src/bf16_gemm.rs` | ~490 | BF16 GEMM microkernel |
| `rustyblas/src/int8_gemm.rs` | ~805 | INT8 GEMM with VNNI |

---

## 10. Testing

```bash
# Full test suite
cargo test

# Core only (SIMD, kernels, hybrid, BF16)
cargo test -p rustynum-core

# Arrow bridge (horizontal sweep, indexed cascade)
cargo test -p rustynum-arrow

# BLAS (GEMM, level1-3)
cargo test -p rustyblas

# Python bindings
cd bindings/python && cargo test

# Lint — run often, catches type errors and unused imports early
cargo clippy --workspace -- -D warnings

# Miri — catches UB in split_at_mut / raw pointer patterns
# ALWAYS use timeout — without it Miri runs 1-3 hours on full workspace
timeout 300 cargo miri test -p rustynum-core   # 5 min cap
timeout 300 cargo miri test -p rustynum-holo   # 5 min cap
# NOTE: rust-toolchain.toml pins nightly because rustyblas, rustymkl,
# rustynum-rs use #![feature(portable_simd)]. This is ergonomics — std::arch
# produces identical machine code. See P2 TODO for the stable port plan.
```

### Test Counts (2026-02-25)

| Crate | Tests | Status |
|-------|-------|--------|
| rustynum-core | 131 | 0 failures |
| rustynum-arrow | 44 | 0 failures |
| rustyblas | ~50 | 0 failures |
| Total | ~225 | All passing |

---

## 11. Anti-Patterns — DO NOT

- **DO NOT** import BindSpace, crewai-rust, n8n-rs, or neo4j-rs
- **DO NOT** do IO (file, network, database) in rustynum functions
- **DO NOT** copy between Overlay and Fingerprint<256> — they ARE the same memory, use `as_fingerprint_words()`
- **DO NOT** use `Arc<Mutex>` for parallel output — use `split_at_mut` or XOR delta layers
- **DO NOT** use `&self → &mut` (UB) — use `&mut self` with field borrows or delta layers
- **DO NOT** use `is_x86_feature_detected!()` in hot loops — use dispatch
- **DO NOT** hardcode AVX-512 in test assertions — use conditional
- **DO NOT** add `jitson` to workspace members or use `exclude`
- **DO NOT** delete archive crates
- **DO NOT** add new nightly features — `portable_simd` is the ONLY nightly dep (P2 TODO to port to std::arch).
  The whole stack targets stable Rust 1.93, Arrow 57, DataFusion 51. `rust-toolchain.toml` pins nightly
  only because of portable_simd in rustyblas/rustymkl/rustynum-rs. Other repos (ladybug-rs, crewai-rust,
  n8n-rs) build on stable. Do NOT add new `#![feature(...)]` attributes.
- **DO NOT** store intermediate BF16 values — accumulate in FP32
- **DO NOT** use dynamic SKU sizing — K0/K1/K2 are fixed at 16K or 64K
- **DO NOT** use `RefCell`, `UnsafeCell`, or runtime borrow checks — the algebra handles isolation

---

## 12. The Lance Zero-Copy Contract (HARD-WON — 6 REWRITES)

> **After 1.5M lines of code and 6 rewrites across sessions, this lesson was
> learned the hard way. DO NOT UNLEARN IT.**

### The Insight

**Arrow is the zero-copy computational backbone. Lance is a persistence layer (cold tier).**

When rustynum is wired into ladybug-rs through Lance:
- Lance mmap's data into Arrow `Buffer`s (64-byte aligned)
- rustynum's `Blackboard` allocations are 64-byte aligned
- `Fingerprint<256>` is `[u64; 256]` = 2048 bytes, same as `AlignedBuf2K`
- **They are pointer-compatible. NEVER copy between them.**

### The 3 Breaks That Were Found (and must not recur)

| Break | Location | What Went Wrong | Fix Applied |
|-------|----------|----------------|-------------|
| **Arrow → CogRecord copies** | `rustynum-arrow/arrow_bridge.rs` | `.to_vec()` per row × 4 channels = 819MB/100K records | P1 debt: needs `CogRecordView<'a>` borrowing variant |
| **Index build copies again** | `rustynum-arrow/indexed_cascade.rs` | `extend_from_slice()` into 4 new Vecs = 819MB more | P1 debt: needs column views over Arrow buffers |
| **Duplicate SIMD in ladybug** | `ladybug-rs/src/core/simd.rs` | Compile-time dispatch, not runtime; reimplements what rustynum already has | Should call `rustynum_core::simd::select_hamming_fn()` |

### The Rule When Wiring rustynum Into Lance

```
DO:   Arrow Buffer → &[u8] pointer reinterpret → rustynum SIMD kernels
DO:   FingerprintBuffer.get(i) → &[u64; 256] directly into mmap'd buffer
DO:   Overlay.as_fingerprint_words() → &[u64; 256] zero-copy view
DO:   Use select_hamming_fn() for runtime-dispatched SIMD (one impl, all CPUs)

DON'T: .to_vec() on Arrow columns
DON'T: NumArrayU8::new(arrow_slice.to_vec()) — this copies
DON'T: Build a second SIMD implementation alongside rustynum's
DON'T: Use compile-time #[cfg(target_feature)] for SIMD — use runtime dispatch
```

### What's Still Not Wired (Acceleration Stack)

These rustynum crates are battle-tested but NOT yet connected to the Lance data path:

| Crate | What It Has | Where It Should Wire |
|-------|-------------|---------------------|
| **rustyblas** | 138 GFLOPS GEMM, INT8 VNNI, BF16 mixed-precision | `rustynum-arrow` for batch similarity, projection |
| **rustymkl** | VML (exp, ln, sin, sqrt), LAPACK, FFT | NARS truth scoring (sigmoid), spectral analysis |
| **jitson** | JSON config → native AVX-512 scan kernels | Per-query compiled scans on Arrow buffers |
| **rustynum-core/kernels** | K0/K1/K2 cascade | Already wired via indexed_cascade |
| **rustynum-core/hybrid** | Tiered dispatch pipeline | Wired for search, not for bulk ingest |

### Cross-Repo References

- `ladybug-rs/CLAUDE.md` § "The Rustynum Acceleration Contract"
- `ladybug-rs/docs/LANCE_HARVEST.md` — why Lance is cold-tier only
- `ladybug-rs/docs/BINDSPACE_UNIFICATION.md:1122` — the `Buffer::from_slice_ref()` warning
- `ladybug-rs/PLAN-RUSTYNUM-INTEGRATION.md` — 5-phase integration roadmap
- `crewai-rust/CLAUDE.md` § "Storage Strategy"
- `n8n-rs/CLAUDE.md` § "Arrow Zero-Copy Chain"

---

## 13. OPEN TODOs — Wiring Checklist (SESSION-DURABLE)

> **READ THIS EVERY SESSION.** These are the concrete tasks that remain.
> Do NOT invent new code. Wire EXISTING acceleration. Mark items DONE with
> date when completed. If you skip an item, explain why in a comment.

### P0 — Zero-Copy Breaks (must fix before any new features)

- [ ] **CogRecordView<'a>** — Create borrowing variant of CogRecord in `rustynum-arrow/src/arrow_bridge.rs`
  - Currently: `record_batch_to_cogrecords()` (line ~114) calls `.to_vec()` 4× per row = 819MB/100K records
  - Fix: Add `CogRecordView<'a> { meta: &'a [u8], cam: &'a [u8], btree: &'a [u8], embed: &'a [u8] }`
  - Then: `record_batch_to_cogrecord_views()` returns `Vec<CogRecordView<'a>>` borrowing from Arrow
  - File: `rustynum-arrow/src/arrow_bridge.rs:114-147`

- [ ] **CascadeIndices::build_from_arrow()** — Eliminate second copy in `rustynum-arrow/src/indexed_cascade.rs`
  - Currently: `build()` (line ~80) calls `extend_from_slice()` 4× into new Vecs = another 819MB
  - Fix: Add `build_from_arrow(meta: &[u8], cam: &[u8], btree: &[u8], embed: &[u8])` that takes raw column bytes
  - Use `arrow_to_flat_bytes()` from `datafusion_bridge.rs` to get `&[u8]` from Arrow columns
  - File: `rustynum-arrow/src/indexed_cascade.rs:80-108`

- [ ] **Delete ladybug-rs/src/core/simd.rs** — 348 lines duplicating rustynum's SIMD dispatch
  - Currently: ladybug-rs has its own `hamming_distance()`, `hamming_avx512()`, `hamming_avx2()`, `HammingEngine`
  - Fix: Delete file, replace all call sites with `rustynum_core::simd::select_hamming_fn()`
  - Blocker: rustynum's simd.rs uses `portable_simd` (nightly). ladybug-rs builds on stable.
  - Workaround: Use `rustynum_core::simd::hamming_distance()` which has scalar fallback
  - Also fix: `ladybug-rs/src/storage/bind_space.rs:1848-1855` — scalar hamming loop, replace with rustynum call
  - Also fix: `ladybug-rs/src/core/rustynum_accel.rs:148` — `.to_vec()` in container_bundle

### P1 — Wire Existing Acceleration Into Data Path

- [ ] **rustyblas GEMM for batch similarity** — After cascade filtering, survivors need pairwise distance
  - Entry point: `rustyblas::level3::sgemm()` (138 GFLOPS)
  - Wire into: `rustynum-arrow/src/horizontal_sweep.rs` for batch post-filtering
  - Input: Survivors from K0/K1/K2 as `&[f32]` slices (quantized from fingerprints)
  - Use Blackboard for aligned memory: `alloc_f32("A", n) + alloc_f32("B", n) + alloc_f32("C", n)`

- [ ] **rustymkl VML for NARS truth scoring** — sigmoid, exp, log on truth values
  - Entry point: `rustymkl::vml::vsexp()` — vectorized exp on `&[f32]`
  - Wire into: ladybug-rs NARS truth computation (post-search confidence scoring)
  - Input: Truth frequency/confidence arrays

- [ ] **jitson JIT scan for per-query kernels** — Compile search config to native code
  - Entry point: `jitson::JitEngine::compile_scan(params) -> ScanKernel`
  - Wire into: `rustynum-arrow/src/horizontal_sweep.rs` — add `jit_kernel: Option<ScanKernel>` to config
  - Input: JSON/YAML thinking style → `ScanParams { threshold, top_k, prefetch_ahead, focus_mask }`
  - Already connected from crewai-rust: `src/persona/jit_link.rs` produces `JitScanParams`
  - NOTE: jitson is NOT a workspace member (depends on wasmtime Cranelift fork). Build separately.

- [ ] **Wire rustynum_accel into core search** — Currently only called from Python FFI
  - `ladybug-rs/src/core/rustynum_accel.rs` has `fingerprint_hamming()`, `container_hamming()`, `slice_hamming()`
  - These call rustynum's runtime-dispatched SIMD — but only used in `python/mod.rs:39`
  - Wire into: `ladybug-rs/src/storage/bind_space.rs` search functions

### P2 — Toolchain & Build

- [ ] **Port portable_simd to std::arch** — Ergonomics only, NOT performance
  - ~879 call sites across: rustyblas (158), rustymkl (198), rustynum-rs (523)
  - Mechanical: `Simd::<f32,16>::from_slice(a)` → `_mm512_loadu_ps(a.as_ptr())`
  - Mechanical: `.reduce_sum()` → `_mm512_reduce_add_ps()`
  - Mechanical: `Simd::splat(x)` → `_mm512_set1_ps(x)`
  - std::arch intrinsics produce IDENTICAL machine code — portable_simd is prettier, not faster
  - BF16: Neither path has native bf16 type. Store as u16, widen to f32 for compute.
    AVX-512 BF16 intrinsics (_mm256_dpbf16_ps) are std::arch on stable anyway.
  - Stable alternatives: `wide` crate, `pulp` crate, or manual std::arch
  - After port: change `rust-toolchain.toml` from `nightly` to `channel = "1.93"`
  - Priority: LOW — nightly works fine, this is code quality not performance

- [ ] **Fix qualia_xor crate** — Missing candle_core, candle_nn, tokenizers dependencies
  - 7 compilation errors: unresolved imports
  - Either add deps to Cargo.toml or gate behind feature flag
  - Currently excluded from CI (`--exclude qualia_xor`)

### DONE

<!-- Move completed items here with date -->
<!-- Example: - [x] 2026-02-27: Added §12 Lance Zero-Copy Contract to CLAUDE.md -->
- [x] 2026-02-27: DeltaLayer + LayerStack + CollapseGate implemented in rustynum-core
- [x] 2026-02-27: Overlay + AlignedBuf2K + MultiOverlay in rustynum-holo
- [x] 2026-02-27: as_fingerprint_words() zero-copy bridge
- [x] 2026-02-27: split_at_mut / parallel_into_slices (Arc<Mutex> eliminated)
- [x] 2026-02-27: Blackboard takes &mut self (UB fixed)
- [x] 2026-02-27: NumElement supertrait
- [x] 2026-02-27: cascade_scan_4ch() zero-copy via arrow_to_flat_bytes()
- [x] 2026-02-27: CI workflows with lint + miri (5 min timeout per crate)

---

*This document governs rustynum development. Read
[ada-docs/architecture/FOUR_LEVEL_ARCHITECTURE.md](https://github.com/AdaWorldAPI/ada-docs/blob/main/architecture/FOUR_LEVEL_ARCHITECTURE.md)
for the cross-repo architectural contract.*

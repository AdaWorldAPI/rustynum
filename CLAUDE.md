# CLAUDE.md — rustynum

> **Last Updated**: 2026-02-25
> **Branch**: `claude/vsaclip-hamming-recognition-y0b94`
> **Owner**: Jan Hübener (jahube)

---

## READ THIS FIRST — Role in the Architecture

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
Tier 4: AMX Tiles / GPU                        — NOT YET IMPLEMENTED
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
The CollapseGate is the airlock between superposition and ground truth:

```
writers produce deltas → superposition exists (bundle of coexisting deltas)
                              │
                         CollapseGate (airlock)
                              │
                    ┌─────────┼─────────┐
                  FLOW       HOLD      BLOCK
                    │         │         │
               collapse    keep      discard
               to ground   super-    super-
               truth       position  position
```

- **Before the gate**: superposition. Deltas coexist. No race — ground is `&self`.
- **FLOW**: gate opens, superposition collapses into ground truth (commit).
- **HOLD**: gate stays closed. Superposition persists. More evidence accumulates.
- **BLOCK**: contradiction detected. Superposition discarded. Ground unchanged.

The gate evaluates conflict via AND + popcount on the deltas themselves.
XOR is the algebra. Bundle is the superposition. The gate is the decision.

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
| Nightly | ONLY for testing AMX; upstream uses stable 1.93 |

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

# Python bindings (requires nightly for PyO3)
cd bindings/python && cargo test
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
- **DO NOT** use nightly features in non-test code (upstream = stable 1.93)
- **DO NOT** store intermediate BF16 values — accumulate in FP32
- **DO NOT** use dynamic SKU sizing — K0/K1/K2 are fixed at 16K or 64K
- **DO NOT** use `RefCell`, `UnsafeCell`, or runtime borrow checks — the algebra handles isolation

---

*This document governs rustynum development. Read `/home/user/CLAUDE.md`
for the cross-repo architectural contract.*

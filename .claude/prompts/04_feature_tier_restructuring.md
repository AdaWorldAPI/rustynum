# Rustynum Feature Tier Restructuring

## The Model

Three tiers. Additive. Clean.

```
rustynum (default)        — full compute stack
rustynum + "qualia"       — + felt/semantic layer
rustynum + "holo"         — + experimental wave substrate
```

### Tier 1: Default (no feature flag needed)

Everything that is numerical compute. If it's math, it's here.

```
Arrays, SIMD runtime dispatch (AVX-512 → AVX2 → scalar)
BLAS (rustyblas INT8 GEMM, FP32/FP64 GEMM)
MKL-style vectorized math (rustymkl exp/sigmoid/tanh)
BNN inference (rustynum-bnn — binary neural net forward pass)
SPO encode/distance/harvest (three XORs, three popcounts, cross-plane vote)
CLAM tree search (rustynum-clam — metric space partitioning, CAKES k-NN)
CAM locate (content-addressable Hamming nearest neighbor)
NARS truth arithmetic (frequency/confidence, revision rule, inference rules)
Arrow/Lance zero-copy bridge (rustynum-arrow)
Stripe shift detector (5 CMP instructions on popcount output)
AccumulatedHarvest (EMA + NARS revision across searches)
ClamPath encoding (u16 bipolar split path)
```

**Rationale**: SPO distance is three popcounts. CLAM is a B-tree. NARS is arithmetic. These are numerical primitives with cognitive names. Removing them would cripple rustynum — they're free and they make every search operation produce typed structural information as a byproduct.

### Tier 2: `qualia` (opt-in)

The felt/semantic layer. Researchers doing clean numerical benchmarks may not want this.

```
Compression (panCAKES XOR-diff from cluster centers)
CAM hydration (semantic content → fingerprint population)
qualia_xor (qualia-space XOR operations)
```

**Rationale**: These modules interpret numerical results as subjective experience. Not everyone wants their distance metric to have an opinion about what things feel like. Clean research needs clean numbers.

### Tier 3: `holo` (opt-in, experimental)

Holographic memory and wave substrate. Unfinished.

```
rustynum-holo (phase-space holographic operations)
rustynum-oracle (three-temperature oracle parameter discovery)
Signed quinary wave substrate (when implemented)
rustynum-carrier (frozen archive — carrier waveform model)
rustynum-focus (frozen archive — focus-gating model)
```

**Rationale**: This is the experimental frontier. The wave interference physics, Berge 3-hypergraph factorization, and signed quinary INT8 resonator are designed but not implemented. Feature-gating keeps it from polluting stable builds.

## Pinned Dependencies

These are infrastructure, not optional:

```toml
lancedb = "=2.0.0"
arrow = "=57"
datafusion = "=51"

[package]
rust-version = "1.93"
```

Arrow/Lance/DataFusion are the memory substrate — like libc. BindSpace zero-copy requires them. ~50MB compile footprint, one-time cost, stable foundation.

## Cargo.toml Structure

```toml
[features]
default = []
# Default includes the FULL compute stack:
#   arrays, SIMD, BLAS, MKL, BNN, SPO, CLAM, CAM, NARS, Arrow/Lance
#   These are ALL numerical primitives — XOR, popcount, tree search, truth arithmetic
#   Python users get this via pip install. No feature flags needed.

qualia = []
# Adds: compression (panCAKES), CAM hydration, qualia_xor
# The felt/semantic layer — opt-in for those who want it

holo = []
# Adds: holographic phase-space, wave substrate, oracle
# Experimental and unfinished — opt-in only
```

## Consumer Examples

```toml
# Python user / clean researcher — fast arrays + SPO + CLAM + NARS
rustynum = { path = "../rustynum" }

# ladybug-rs — everything including felt layer and experimental
rustynum = { path = "../rustynum", features = ["qualia", "holo"] }

# Someone benchmarking SPO distance vs cosine — just default
rustynum = { path = "../rustynum" }
```

## The Boundary Rule

**If it's compute, it's default. If it's meaning, it's `qualia`. If it's unfinished, it's `holo`.**

- `spo_distance()` → compute (three popcounts) → default
- `harvest_to_nars()` → compute (popcount ratios) → default  
- `AccumulatedHarvest` → compute (EMA + revision) → default
- CLAM tree → compute (metric partitioning) → default
- CAM locate → compute (Hamming NN) → default
- ClamPath → compute (u16 encoding) → default
- Stripe shift → compute (5 CMPs) → default
- panCAKES compression → meaning (semantic-aware diff) → `qualia`
- CAM hydration → meaning (content → felt fingerprint) → `qualia`
- qualia_xor → meaning (qualia-space operations) → `qualia`
- Holographic phase ops → unfinished → `holo`
- Wave substrate → unfinished → `holo`
- Oracle → unfinished → `holo`

## What This Means for Current Crates

```
STAYS DEFAULT (no gate):
  rustynum-rs          — core arrays + SIMD
  rustynum-core        — shared primitives + blackboard + SigmaGate + NARS types
  rustynum-bnn         — BNN inference + SPO types + CrossPlaneVote + CausalSaliency
  rustynum-arrow       — Arrow/Lance zero-copy bridge
  rustynum-clam        — CLAM tree + CAKES search + panCAKES (basic)
  rustyblas            — BLAS (INT8 GEMM, matmul)
  rustymkl             — vectorized math
  jitson               — JIT (own workspace, unaffected)

FEATURE-GATED "qualia":
  qualia_xor           — qualia XOR operations
  Parts of rustynum-clam involving semantic compression/hydration

FEATURE-GATED "holo":
  rustynum-holo        — holographic phase-space
  rustynum-oracle      — three-temperature oracle
  rustynum-carrier     — frozen archive
  rustynum-focus       — frozen archive

FROZEN ARCHIVES (keep, no feature gate, just don't compile by default):
  rustynum-archive     — v2 snapshot
  rustynum-archive-v3  — v3 snapshot
```

## Implementation

1. Add `qualia` and `holo` features to workspace `Cargo.toml`
2. Gate `qualia_xor` crate compilation on `qualia` feature
3. Gate `rustynum-holo`, `rustynum-oracle`, `rustynum-carrier`, `rustynum-focus` on `holo` feature
4. Ensure default build (no features) compiles and passes all tests
5. Ensure `--all-features` compiles and passes all tests
6. Update ladybug-rs dependency to `features = ["qualia", "holo"]`
7. Exclude archive crates from default workspace members

## Verification

```bash
# Default — full compute stack, no felt/experimental
cargo test
cargo test --no-default-features  # same thing (default = [])

# With qualia
cargo test --features qualia

# Everything
cargo test --all-features

# Check no qualia/holo code leaks into default
cargo build 2>&1 | grep -i "qualia_xor\|holo\|oracle\|carrier\|focus"  # should be empty
```

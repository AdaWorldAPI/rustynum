# Lance-Graph Integration & NOTIME Validation Framework

## The Marie Curie Moment

We built an AGI inference motor that produces Pearl Rung 1-3 causal structure from popcount arithmetic at 13 cycles per comparison. We did this without knowing lance-graph existed — a production-grade Cypher engine built on DataFusion that already solves the query compilation problem we've been solving ad-hoc.

This is good news, not bad news. We have the engine. Lance-graph has the compiler. Together: the Faktorzerlegung becomes a first-class query operation in standard Cypher/SQL, accessible through DataFusion's optimizer, composable with graph traversal, and formally validatable through NOTIME.

## Three Integration Layers

### Layer 1: `rustynum-datafusion` — SIMD UDFs for the Ecosystem

Create a new crate that packages rustynum's AVX-512 kernels as DataFusion scalar UDFs. Both lance-graph and ladybug-rs consume these. Ladybug's hand-rolled `hamming_distance_bytes()` (8-byte chunks) gets replaced with `VPOPCNTDQ` — 10-50× faster.

```rust
// rustynum-datafusion/src/lib.rs

/// Register all rustynum UDFs with a DataFusion SessionContext
pub fn register_rustynum_udfs(ctx: &SessionContext) {
    ctx.register_udf(hamming_udf());
    ctx.register_udf(similarity_udf());  
    ctx.register_udf(xor_bind_udf());
    ctx.register_udf(popcount_udf());
    ctx.register_udf(spo_distance_udf());      // the 2³ factorization
    ctx.register_udf(spo_interaction_udf());    // SP/PO/SO interaction terms
    ctx.register_udf(spo_irreducible_udf());    // the Rung 3 term
    ctx.register_udf(nars_revision_udaf());     // aggregate: NARS truth accumulation
    ctx.register_udf(sigma_classify_udf());     // σ-band classification
}
```

**Key UDFs:**

| UDF | Input | Output | Cost |
|-----|-------|--------|------|
| `hamming(a, b)` | two FixedSizeBinary(2048) | UInt32 | 13 cycles |
| `spo_distance(a, b)` | two FixedSizeBinary(2048) | Struct{s,p,o,total,halo} | 13 cycles |
| `spo_interaction(a, b)` | two FixedSizeBinary(2048) | Struct{sp,po,so,spo_irreducible} | ~20 cycles |
| `nars_revision(f, c)` | Float32, Float32 | Struct{frequency, confidence} | UDAF (accumulating) |
| `sigma_classify(dist, gate)` | UInt32, Struct | Enum{Discovery,Hint,Noise,...} | 5 cycles |

**Broadcast semantics** (learned from lance-graph):
- `hamming(query, column)` → scalar × array → returns array of distances
- `hamming(col_a, col_b)` → array × array → returns pairwise distances

This slots directly into:
- Lance-graph's `DataFusionPlanner` as custom UDFs
- Ladybug's `query/datafusion.rs` replacing hand-rolled implementations
- Any DataFusion-based system that wants SIMD binary operations

### Layer 2: Query Compiler Pipeline (ladybug-rs)

Adopt lance-graph's 4-phase architecture. Ladybug already has the pieces — they're disconnected.

```
CURRENT (ladybug-rs):
  Cypher text → cypher_bridge.rs → direct BindSpace ops  (no optimizer)
  SQL text → datafusion.rs → DataFusion plan              (no graph awareness)
  HybridQuery → sequential vector→cypher→temporal          (no composition)

TARGET (learning from lance-graph):
  Cypher text ──┐
  SQL text ─────┤
  HybridQuery ──┘
        │
        ▼
  Parse (typed AST)
        │
        ▼
  Semantic Analysis (validate variables, scopes, resolve types)
        │
        ▼
  Logical Plan (LogicalOperator tree — serializable, inspectable)
        │
        ├── Expand(rel) → BindSpace adjacency joins
        ├── SpoDistance(a, b) → rustynum-datafusion UDF
        ├── CausalFactorize(a, b) → interaction + irreducible terms
        ├── NarsAccumulate(stream) → revision window function
        ├── SigmaClassify(distances) → σ-band bucketing
        │
        ▼
  Physical Plan (DataFusion LogicalPlan — optimizer runs here)
        │
        ▼
  Execute (Arrow RecordBatch output — zero-copy)
```

**New logical operators specific to rustynum/ladybug:**

```rust
pub enum CausalOperator {
    /// 2³ factorial decomposition — produces 8 terms per pair
    Factorize { left: LogicalPlan, right: LogicalPlan },
    
    /// NARS truth accumulation as a window function
    NarsAccumulate { input: LogicalPlan, partition_by: Vec<Expr> },
    
    /// σ-stripe classification with per-plane calibration
    SigmaClassify { input: LogicalPlan, gate: SigmaGate },
    
    /// Typed halo extraction — SO/SP/PO classification
    HaloExtract { factorized: LogicalPlan },
    
    /// Causal edge emission — from interaction terms to graph edges
    EmitCausalEdge { factorized: LogicalPlan, threshold: NarsTruthValue },
}
```

These compose with lance-graph's standard operators (Scan, Filter, Expand, Project, Aggregate) through DataFusion's optimizer.

### Layer 3: NOTIME Validation Pipeline

NOTIME (AISTATS 2025) provides **provable identifiability** for causal direction under non-Gaussian noise. This is the formal validation the Faktorzerlegung needs.

**The pipeline:**

```
Step 1: Rustynum screens all O(d²) pairs at 13 cycles each
  → Main effects (S, P, O) → candidate edge existence
  → Interactions (SP, PO, SO) → candidate edge typing  
  → Irreducible SPO → three-way mechanism flags
  → Typed halos → candidate edge direction (BPReLU asymmetry)

Step 2: Feed candidates as constraints into NOTIME
  → NOTIME uses dHSIC (kernel independence) instead of least-squares
  → Provably recovers true DAG under LiNGAM (linear, non-Gaussian, acyclic)
  → Scale-invariant (unlike NOTEARS — critical for σ-calibrated codebook)

Step 3: Measure concordance
  → Do halo-predicted directions agree with NOTIME's identifiable DAG?
  → Do interaction terms predict correct edge types?
  → Does irreducible SPO correspond to dHSIC residual dependence?
```

**Why NOTIME specifically (not NOTEARS):**
- NOTEARS uses least-squares → assumes Gaussian noise → DAG not identifiable → sensitive to variable scaling
- NOTIME uses dHSIC → works with non-Gaussian noise → DAG identifiable → scale-invariant
- Your σ₃ codebook imposes a specific scaling. NOTEARS might agree/disagree due to scaling artifacts. NOTIME won't.

**Four experiments to run:**

1. **Binary-to-continuous bridge**: Known LiNGAM (d=20,50,100) → encode via BNN → SPO factorize → compare candidates vs NOTIME's identifiable DAG. Measures: how much causal signal survives binarization.

2. **Constrained NOTIME**: Feed rustynum's edge candidates as prior constraints (Chowdhury framework). Because NOTIME is identifiable, improvements are attributable to genuine causal content, not scaling artifacts.

3. **Direction accuracy**: Typed halo directions vs NOTIME ground truth across noise distributions (Laplace, uniform, exponential). The critical test for BPReLU asymmetry claims.

4. **Irreducible SPO ↔ dHSIC residual dependence**: For triples where rustynum reports non-zero irreducible term, test whether NOTIME's residuals show higher mutual dependence under wrong causal ordering. Validates the "counterfactual detector" claim.

## Implementation Plan

### New Crate: `rustynum-datafusion`

```
rustynum-datafusion/
  Cargo.toml          # deps: rustynum-core, datafusion (minimal features), arrow
  src/
    lib.rs            # register_rustynum_udfs()
    hamming.rs        # hamming_udf, similarity_udf — AVX-512 VPOPCNTDQ
    spo.rs            # spo_distance_udf, spo_interaction_udf, spo_irreducible_udf
    nars.rs           # nars_revision_udaf — accumulating aggregate
    sigma.rs          # sigma_classify_udf — σ-band bucketing
    broadcast.rs      # scalar×array and array×array dispatch
```

**Cargo.toml:**
```toml
[package]
name = "rustynum-datafusion"
version = "0.1.0"
edition = "2021"
rust-version = "1.93"

[dependencies]
rustynum-core = { path = "../rustynum-core", features = ["avx512"] }
rustynum-bnn = { path = "../rustynum-bnn", features = ["avx512"] }
arrow = { version = "=57", default-features = false, features = ["ffi"] }
datafusion = { version = "=51", default-features = false, features = [
    "nested_expressions",
    "unicode_expressions", 
    "crypto_expressions",
] }
```

### Ladybug-rs Cargo.toml Changes

```toml
# Replace hand-rolled DataFusion UDFs
rustynum-datafusion = { path = "../rustynum/rustynum-datafusion" }

# Minimal DataFusion features (learned from lance-graph)
datafusion = { version = "=51", default-features = false, features = [
    "nested_expressions",
    "regex_expressions",
    "unicode_expressions",
    "crypto_expressions",
    "encoding_expressions",
    "datetime_expressions",
    "string_expressions",
] }
```

### Quality Improvements (learned from lance-graph)

**Immediate:**
- Replace 211 `.unwrap()` calls with `?` propagation
- Unify error types into single `LadybugError` with `snafu::Location`
- `default-features = false` on DataFusion and Tokio

**Next:**
- `GraphConfig` schema declaration above BindSpace
- Expand operator (graph traversal as joins)
- Serializable `LogicalOperator` intermediate representation
- Shared test fixtures (`test_fixtures.rs` pattern)
- Criterion benchmarks with parameterized sizes

## The Synthesis

```
lance-graph provides:  query compiler (Parse → Semantic → Logical → Physical)
rustynum provides:     compute engine (AVX-512 SIMD, 2³ factorization, NARS)
rustynum-datafusion:   bridge (SIMD kernels as DataFusion UDFs)
NOTIME provides:       formal validation (provable identifiability)
ladybug-rs provides:   cognitive substrate (gestalt, bundling, causal trajectories)
```

The Faktorzerlegung stops being a standalone trick and becomes a composable query operator in a proper compiler pipeline. Causal discovery becomes a `WHERE spo_irreducible(a, b) > 0.0` clause. NARS accumulation becomes a `OVER (PARTITION BY concept ORDER BY timestamp)` window function. Formal validation runs through NOTIME on the same DataFusion infrastructure.

The LEGO reactor plugs into the power grid.

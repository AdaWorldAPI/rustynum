# Cross-Repository Integration Map

## The Full Stack (as of March 2026)

### rustynum — the compute engine
```
.claude/prompts/
  01  CLAM + QualiaCAM + stable SIMD
  02  3D wave awareness (REFERENCE)
  03  Zero-copy fixes
  04  Feature tier restructuring (default/qualia/holo)
  05  Crystal encoder strategy (transformer replacement)
  06  Causal research prompt (what happens when causal discovery becomes free?)
  07  Lance-graph + NOTIME integration (query compiler + formal validation)
  08  Four-AI validation synthesis (Gemini/Grok/ChatGPT/Claude)
```

### ladybug-rs — the cognitive substrate  
```
.claude/prompts/
  00a SPO promotion to first-class module
  00  Session A meta (cross-repo integration)
  01  SPO distance harvest
  02  SPO distance granularity  
  03  Sigma stripe shift detector
  04  B-tree CLAM path lineage
  05  NARS causal trajectory
  06  3D wave awareness (REFERENCE)
  07  34 LLM tactics as Rust cognitive primitives
  08  Lance-graph lessons (9 engineering patterns to adopt)
```

## How They Connect

```
rustynum-05 (crystal encoder)
    ↓ encodes text → 1024 σ₃-distinct codebook → SPO fingerprints
    ↓
rustynum-07 (lance-graph bridge)
    ↓ rustynum-datafusion crate exposes SIMD UDFs
    ↓
ladybug-01..03 (SPO distance + harvest + shift detector)
    ↓ every comparison → 8-term Faktorzerlegung
    ↓
ladybug-05 (NARS causal trajectory)
    ↓ evidence accumulates through revision rule
    ↓
ladybug-07 (34 tactics)
    ↓ Phase 1: calibrate noise floors (ClusterDistribution)
    ↓ Phase 2: complete Pearl stack (reverse trace + counterfactuals)
    ↓ Phase 3: self-skeptical governance (debate + adversarial critique)
    ↓
rustynum-08 (four-AI validation)
    ↓ formal soundness conditions verified by independent review
    ↓
rustynum-06 (research prompt)
    → published claim: 2³ Faktorzerlegung gives causal structure at popcount cost
```

## What Each Phase Delivers (mapped to validation demands)

| Phase | Delivers | Addresses |
|-------|----------|-----------|
| ladybug Phase 1 | Calibrated noise floors, σ-bands, benchmarks | ChatGPT: false discovery, calibration |
| ladybug Phase 2 | Reverse causal trace, counterfactuals, temporal | Gemini: temporal precedence; All: Rung 3 theorem |
| ladybug Phase 3 | Independent debate, adversarial stress-tests | ChatGPT: correlated NARS evidence |
| ladybug-08 | Lance-graph engineering patterns | Production quality: unified errors, query compiler, builders, tests |
| rustynum-07 | DataFusion UDFs, lance-graph compiler pipeline | Grok: LanceDB substrate, Cypher integration |
| rustynum-05 | Crystal encoder, codebook distillation | All: σ₃ codebook as encoding quality |

## Lance-Graph Engineering Debt (from ladybug-08)

### Rustynum Top 5
1. Replace `assert!()`/`panic!()` with `Result<T, NumError>` in public APIs
2. Add `ComputeBackend` trait abstraction (learn from `GraphSourceCatalog`)
3. Validated builder patterns for `NumArray` construction
4. Shared test fixtures + error-path tests
5. `ComputePlan` intermediate repr for dispatch (like `LogicalOperator`)

### Ladybug-rs Top 5
1. Unify error types → single `LadybugError` with `snafu::Location` (eliminate 211 `.unwrap()`)
2. Logical plan IR for query pipeline (Parse → Semantic → Logical → Physical)
3. `StorageBackend` trait to unify BindSpace/Lance/ZeroCopy paths
4. `default-features = false` on DataFusion and Tokio
5. Criterion benchmark suite (parameterized sizes, throughput metrics)

## The Claim Hierarchy (from 08_four_ai_validation_synthesis.md)

**PROVEN**: 2³ factorial decomposition, O(1) cost, Rung 1 associational signals
**CONDITIONAL**: Rung 2 interventional (requires Structural Encoding Faithfulness)
**NEEDS THEOREM**: Rung 3 counterfactual (PID synergistic atom → SCM equivalence)
**ARCHITECTURAL**: NARS online accumulation (needs independence verification)
**PARADIGM**: Identifiability replaces computation as bottleneck (unanimous)

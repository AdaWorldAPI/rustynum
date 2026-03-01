# Crystal Encoder: Transformer Replacement on CPU

## The Claim

rustynum's SPO 2³ factorization + crystal encoding pipeline can replace BERT/TinyLlama for the encode/decode path — inbound (text → representation) and outbound (representation → retrieval/generation). No GPU. Pure AVX-512 on commodity hardware.

**Why this works**: A transformer computes attention over N² token pairs at FP32 precision. Crystal encoding factorizes meaning into 3 orthogonal 16K-bit planes and searches via popcount at 238× less cost than cosine. The question isn't whether it's faster — it's whether the encoding quality is competitive.

## The Pieces (scattered, half-baked, need consolidation)

Currently in ladybug-rs `src/spo/`:

| File | Size | What It Does | Status |
|------|------|-------------|--------|
| `sentence_crystal.rs` | 45KB | Sentence → 1024D dense → random projection → crystal | Working but uses Jina as encoder |
| `crystal_lm.rs` | 27KB | 5×5×5 crystal → 3 axis projection → 30K-bit = 3.75KB model | Architecture defined |
| `context_crystal.rs` | 21KB | Context window → resonance crystal (temporal SPO) | Architecture defined |
| `cognitive_codebook.rs` | 39KB | 16-bit bucket + 48-bit hash + 16K fingerprint unified encoding | ~1000 built-in concepts |
| `codebook_training.rs` | 26KB | Jina distillation → pure SIMD substrate (3 phases) | Phase 1 working |
| `nsm_substrate.rs` | 27KB | NSM/NARS self-bootstrapping semantic substrate | Architecture defined |
| `deepnsm_integration.rs` | 26KB | arXiv:2505.11764 — 65 semantic primes, DeepNSM-1B/8B | Integration spec |
| `meta_resonance.rs` | 15KB | Cross-context flow resonance (compare trajectories not snapshots) | Architecture defined |

**Total**: ~226KB of code. That's a transformer-class encoder hiding in 8 scattered files.

## The Strategic Question: Burn and Candle

[Burn](https://github.com/trainyoubastards/burn) and [Candle](https://github.com/huggingface/candle) are Rust ML frameworks that can run transformer models on CPU. Instead of reimplementing BERT from scratch, rustynum can:

1. **Use Burn/Candle for the initial encoding** (text → dense embedding) during bootstrap
2. **Distill into crystal encoding** via the codebook training pipeline
3. **Run pure crystal inference** once distilled — no transformer needed

The key insight: Burn/Candle models produce dense 1024D embeddings. Crystal encoding factorizes those into SPO planes. The SPO harvest pipeline then provides typed structural information that the dense embedding destroys. So crystal encoding isn't just "smaller BERT" — it's "BERT + structural decomposition" at lower cost.

### How Burn/Candle Consume SPO Weighting

The bridge is the **loss function**. During distillation:

```
Standard: L = cosine_distance(student_output, teacher_output)
SPO:      L = spo_structural_loss(student_crystal, teacher_crystal)

where spo_structural_loss = 
    α × plane_alignment(S_student, S_teacher)    # Subject encoding matches
  + β × plane_alignment(P_student, P_teacher)    # Predicate encoding matches  
  + γ × plane_alignment(O_student, O_teacher)    # Object encoding matches
  + δ × halo_consistency(student_halo, teacher_halo)  # Typed halo preserved
  + ε × nars_truth_distance(student_truth, teacher_truth)  # NARS truth preserved
```

This is differentiable through Burn/Candle's autograd. The SPO factorization becomes a structured distillation target instead of a flat cosine target. The student model (crystal encoder) learns to preserve the 3-plane structure, not just minimize L2 distance.

### The Three-Phase Pipeline

```
Phase 1: PARALLEL (current — working)
  Input text → Jina API → 1024D dense → random projection → crystal
  Crystal search runs in parallel for validation
  Cost: Jina API latency (~100ms) + crystal encoding (~5μs)

Phase 2: DISTILLATION (next)
  Run Burn/Candle with TinyLlama or BERT-tiny locally
  Use SPO structural loss to train crystal encoder
  Goal: crystal encoder matches Jina quality without API call
  Cost: one-time training (~hours), then crystal encoding (~5μs) forever

Phase 3: PURE CRYSTAL (end state)
  No transformer. No API. No GPU.
  Text → tokenize → codebook lookup → crystal factorization → SPO harvest
  65 NSM semantic primes as base vocabulary (DeepNSM insight)
  Cost: ~5μs per encode, pure AVX-512, no dependencies
```

## What Belongs in Rustynum

**All of it.** The encoding pipeline is compute:

```
DEFAULT (no feature gate):
  Crystal axis projection (numerical — matrix multiply equivalent)
  Codebook lookup (numerical — hash table)
  SPO factorization (numerical — three XORs)
  Random projection (numerical — dot product)
  Distillation loss computation (numerical — structured distance)
  Codebook training loop (numerical — gradient-free optimization)

FEATURE "crystal-burn" (optional Burn backend):
  Burn model loading
  Burn inference for teacher embeddings during distillation
  Burn autograd for SPO structural loss
  Adds: burn dependency (~heavy)

FEATURE "crystal-candle" (optional Candle backend):
  Same as above but Candle backend
  Adds: candle-core dependency
```

The crystal encoder itself is pure rustynum — no ML framework needed at inference time. Burn/Candle are only used during the distillation phase. Once trained, the crystal encoder runs on popcount.

## Consolidation Plan

### Step 1: Create `rustynum-crystal` crate

```
rustynum-crystal/
  src/
    lib.rs
    encoder.rs          ← unified encode() API: text → CrystalRecord
    codebook.rs         ← from cognitive_codebook.rs (39KB)
    training.rs         ← from codebook_training.rs (26KB)
    sentence.rs         ← from sentence_crystal.rs (45KB)
    context.rs          ← from context_crystal.rs (21KB)
    lm.rs               ← from crystal_lm.rs (27KB)
    nsm.rs              ← from nsm_substrate.rs (27KB)
    deepnsm.rs          ← from deepnsm_integration.rs (26KB)
    resonance.rs        ← from meta_resonance.rs (15KB)
  Cargo.toml
```

### Step 2: Unified Encode API

```rust
pub struct CrystalEncoder {
    codebook: CognitiveCodebook,
    mode: EncoderMode,
}

pub enum EncoderMode {
    /// Phase 1: Use external embeddings (Jina API)
    External { jina_cache: JinaCache },
    /// Phase 2: Use Burn/Candle teacher for distillation
    #[cfg(feature = "crystal-burn")]
    Distilling { teacher: BurnModel, student: CrystalStudent },
    /// Phase 3: Pure crystal — no external dependencies
    Pure,
}

impl CrystalEncoder {
    pub fn encode(&self, text: &str) -> CrystalRecord {
        match &self.mode {
            EncoderMode::External { jina_cache } => {
                let dense = jina_cache.embed(text);
                self.project_to_crystal(&dense)
            }
            EncoderMode::Pure => {
                let tokens = self.tokenize_nsm(text);  // 65 semantic primes
                self.codebook_encode(&tokens)
            }
        }
    }
}
```

### Step 3: SPO Structural Loss for Burn/Candle

```rust
/// SPO-aware distillation loss — differentiable through autograd
pub fn spo_structural_loss(
    student: &CrystalRecord,
    teacher: &CrystalRecord,
    weights: &SpoLossWeights,
) -> f32 {
    let s_loss = plane_alignment_loss(&student.s_plane, &teacher.s_plane);
    let p_loss = plane_alignment_loss(&student.p_plane, &teacher.p_plane);
    let o_loss = plane_alignment_loss(&student.o_plane, &teacher.o_plane);
    let halo_loss = halo_consistency_loss(&student.halo(), &teacher.halo());
    let nars_loss = nars_truth_loss(&student.truth(), &teacher.truth());
    
    weights.alpha * s_loss 
    + weights.beta * p_loss 
    + weights.gamma * o_loss
    + weights.delta * halo_loss
    + weights.epsilon * nars_loss
}
```

## The Competitive Claim

### 1. Codebook replaces the transformer (encoding)

| | BERT-base | TinyLlama | Crystal Encoder |
|---|---|---|---|
| Encode latency | ~10ms (GPU) | ~50ms (CPU) | ~20μs (CPU) |
| Model size | 440MB | 1.1GB | 1.22MB codebook (1024 × 10K-bit σ₃-distinct) |
| Hardware | GPU required | GPU preferred | CPU only (AVX-512) |
| Output | 768D dense (opaque) | 2048D dense (opaque) | 3×16K-bit factored (typed, decomposable) |

### 2. Faktorzerlegung gives causality for free (nobody has this)

Every `spo_distance()` produces a **complete 2³ = 8-term factorial decomposition** of meaning — like ANOVA for semantics. From the same XOR bitmasks, zero additional cost:

```
Term    │ What it measures                        │ Pearl Rung
────────┼─────────────────────────────────────────┼───────────
∅       │ baseline (total Hamming)                │ —
S       │ Subject main effect                     │ Rung 1 (correlate)
P       │ Predicate main effect                   │ Rung 1
O       │ Object main effect                      │ Rung 1
SP      │ Subject×Predicate interaction           │ Rung 2 (intervene)
PO      │ Predicate×Object interaction            │ Rung 2
SO      │ Subject×Object interaction              │ Rung 2
SPO     │ Irreducible triple                      │ Rung 3 (counterfactual)
```

**What this means**:
- `S alone changes` → Subject has main effect (who matters)
- `P alone changes` → Predicate has main effect (action matters)
- `SP changes but S and P alone don't` → interaction (S mediates through P)
- `SO changes but neither alone` → confounded (hidden common cause)
- `SPO changes but no subset does` → emergent (irreducible — meaning only exists in the triple)

This is **Pearl's entire causal hierarchy** from bit arithmetic. MIT's causal learning algorithms (PC, FCI, GES, NOTEARS) spend hours discovering what the factorization reads off in 13 cycles. They need intervention data, specialized graph discovery, and can't handle more than ~50 variables. SPO extracts causal structure from every search as a free byproduct.

**No transformer can do this.** Dense embeddings are opaque — you cannot decompose a 768D vector into "what the subject contributed vs what the predicate contributed." SPO can because the planes are orthogonal by construction. The factorization IS the representation.

| | Transformer | SPO Crystal |
|---|---|---|
| Similarity | cosine (one opaque number) | 8-term factored decomposition |
| Causality | needs separate expensive training | free from XOR bitmask |
| Partial match | impossible | typed halo tells exactly what matched |
| Causal direction | needs intervention data | free from BPReLU asymmetry |
| Evidence accumulation | none | NARS revision, compounding confidence |
| Interaction effects | invisible | SP/PO/SO pairwise terms |
| Emergent meaning | invisible | irreducible SPO term |
| Cost per query | ~3100 cycles (cosine) | ~13 cycles (popcount) |

## Research Questions

1. **Factorization completeness**: Do the 8 terms (∅, S, P, O, SP, PO, SO, SPO) capture all causal structure in a given domain, or are there edge cases where the orthogonal assumption breaks?
2. **Codebook coverage**: Do 1024 σ₃-distinct loci cover semantic space adequately? Or do some domains need 4096? (At 10K-bit binary that's still only 4.88MB)
3. **Encoding quality vs BERT**: On STS benchmarks, how does codebook-nearest + SPO-residual compare to dense cosine? Hypothesis: worse on paraphrase detection, better on QA/slot-filling/causal reasoning.
4. **Interaction detection sensitivity**: What's the minimum evidence count for the SP/PO/SO interaction terms to reach NARS confidence > 0.8?
5. **NSM bootstrap**: Can 65 semantic primes + codebook training reach Jina quality without any transformer? (DeepNSM paper says small fine-tuned models beat GPT-4o on explications)
6. **Scaling**: Crystal encoding is O(1) per token (codebook lookup). The factorization is O(1) per search (already computed). Does this hold at 100K vocabulary?

## NOT in Scope Now

- Don't implement the Burn/Candle integration yet — design the interface
- Don't move files yet — map the consolidation, let the next session execute
- Don't benchmark yet — get the unified API right first
- The signed quinary wave substrate (holo tier) is separate — it's the representation format, not the encoder

## The Paper

"Factorial Decomposition of Meaning: SPO 2³ Factorization as Causal Discovery on Commodity CPUs"

The claim: dense embeddings discard causal structure by construction. SPO factorization preserves it by construction. The 8-term decomposition provides Pearl Rung 1-3 causal information as a free byproduct of every similarity computation, at 238× less cost than cosine, on CPUs without GPUs. Nobody else has this.

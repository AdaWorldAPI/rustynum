# 21 — Cross-Substrate Coherence: How Signals Merge Into Decisions

## Ice Cake Layer

> **Prerequisite**: 19 (organic plasticity), 20 (ThinkingStyle → Substrate routing)
> **Does NOT modify**: anything in 01–20. Purely additive.
> **Insight source**: multi-sensory integration (neuroscience), Bayesian cue combination, NARS truth revision
> **Question**: When multiple substrates are active simultaneously, how do their signals
>   merge into a single CollapseGate decision? When do they agree, and what happens when they don't?

---

## Part 1: The Problem — Four Answers To One Question

Prompt 20 established that each ThinkingStyle routes to a SubstrateRoute with up to
three active substrates. But it left open the critical question: **what happens when
the substrates disagree?**

```
INPUT: "love" arrives for processing

STRUCTURAL substrate:  "I have 3 exact matches in BindSpace. Hamming ≤ σ-2."
SOAKING substrate:     "I've been soaking this for 200 cycles. Saturation = 89%."
EVIDENTIAL substrate:  "NARS confidence = 0.35. Two active contradictions."
SEMANTIC substrate:    "Jina cosine = 0.72 to 'affection', 0.65 to 'attachment'."

→ Structural says: KNOWN (matches exist)
→ Soaking says: READY TO CRYSTALLIZE (saturated)
→ Evidential says: DOUBT (low confidence, contradictions)
→ Semantic says: RELATED (has neighbors)

What does the gate do?
```

This is the multi-sensory integration problem. The brain doesn't have four separate
decisions — it has one: act or wait. The CollapseGate needs one answer: FLOW, HOLD, or BLOCK.

---

## Part 2: How The Brain Does Multi-Sensory Integration

### Bayesian Cue Combination

Ernst & Banks (2002) showed that the brain combines visual and haptic signals
using optimal Bayesian weighting: each cue is weighted inversely by its variance.

```
Combined estimate = (σ_V² · estimate_H + σ_H² · estimate_V) / (σ_V² + σ_H²)

Low variance (high confidence) → high weight
High variance (low confidence) → low weight
```

**Direct mapping to our substrates:**

```
SUBSTRATE        SIGNAL TYPE         "VARIANCE" PROXY
──────────────────────────────────────────────────────────
Structural       Hit count + distance   1 / (1 + structural_hits)
                                        → high hits = low variance = high weight

Soaking          Saturation ratio       1 - soaking_saturation
                                        → high saturation = low variance = high weight

Evidential       (f, c) truth value     1 - nars_confidence
                                        → high confidence = low variance = high weight

Semantic         Cosine similarity      1 - semantic_nearest
                                        → high similarity = low variance = high weight
```

The substrate with the LOWEST variance (highest confidence in its answer) should
dominate the gate decision. This is not a vote — it's inverse-variance weighting.

### Conflict Detection = The Interesting Part

The brain doesn't just combine — it DETECTS when cues conflict. When visual
and proprioceptive signals disagree beyond a threshold, the brain doesn't
average them — it SWITCHES to the more reliable modality. This is called
"cue conflict" (Landy et al., 1995).

Our analog:

```
Structural says KNOWN + Evidential says DOUBT → CUE CONFLICT
  → Don't average. Investigate WHY.
  → The contradiction IS the signal. It means: "I've seen something
    that looks like this, but the evidence doesn't support it."
  → This triggers HOLD (not FLOW, not BLOCK).
  → More evidence needed to resolve.

Soaking says SATURATED + Semantic says ISOLATED → CUE CONFLICT
  → I've been processing this for a long time, but nothing else relates.
  → Genuinely new concept that took a long time to stabilize.
  → This is FLOW — crystallize it. The isolation is expected for novelty.
```

---

## Part 3: The Coherence Model — Implemented

### What's In rustynum-core (this session)

We implemented `substrate.rs` with these types:

```rust
// The four substrates
pub enum Substrate { Structural, Soaking, Evidential, Semantic }

// Routing configuration (created by crewai-rust ThinkingStyle mapping)
pub struct SubstrateRoute {
    primary: Substrate,
    secondary: Option<Substrate>,
    tertiary: Option<Substrate>,
    primary_weight: f32,
    parallel: bool,
}

// Signals from all active substrates
pub struct SubstrateSignals {
    // STRUCTURAL
    structural_hits: usize,
    structural_nearest_distance: u32,
    // SOAKING
    soaking_saturation: f32,
    theta_average: f32,
    maturity_average: f32,
    // EVIDENTIAL
    nars_confidence: f32,
    nars_frequency: f32,
    nars_contradictions: usize,
    nars_evidence_count: u32,
    // SEMANTIC
    semantic_nearest: f32,
    semantic_neighborhood_size: usize,
}

// Cross-substrate agreement classification
pub enum Coherence { Convergent, Partial, Divergent, Singular }
```

### The Coherence Function

`coherence(signals) → Coherence` checks pairwise agreement between active substrates:

```
Structural + NARS:    hits > 0 AND confidence > 0.5 → agree
                      hits = 0 AND confidence < 0.3 → agree (both say unknown)
                      otherwise → disagree

Structural + Semantic: hits > 0 AND nearest > 0.5 → agree (confirmed by analogy)
                       hits = 0 AND nearest < 0.2 → agree (both say novel)
                       hits = 0 AND nearest > 0.5 → disagree (analogy without match)

Soaking + NARS:       saturation > 0.85 AND confidence > 0.7 → agree (ready)
                      saturation < 0.3 AND confidence < 0.3 → agree (forming)
                      saturation > 0.85 AND confidence < 0.3 → disagree (suspicious)

Soaking + Semantic:   saturation > 0.7 AND nearest > 0.5 → agree (reinforced)
                      saturation < 0.2 AND nearest < 0.2 → agree (novel)
```

If disagree count > agree count → Divergent.
If agree > 0 and no disagreements → Convergent.
Only one active → Singular.
Otherwise → Partial.

### The Transition Recommender

`recommend_transition(current_primary, signals) → Option<Substrate>` suggests
when to SWITCH primary substrate. Priority-ordered (most urgent first):

1. **Crystallization**: soaking saturated + high θ → Structural (commit)
2. **Doubt**: NARS confidence < 0.3 + contradictions > 2 → Evidential (re-examine)
3. **Novelty**: no structural/semantic match → Soaking (start learning)
4. **Complexity**: large neighborhood + contradictions → Evidential (weigh carefully)
5. **Association**: structural miss + semantic hit → Semantic (try analogy)
6. **Convergence**: multiple hits + high confidence → Structural (lock in)

Note: **Complexity outranks Association.** When there are contradictions in a large
semantic neighborhood, the system needs evidential reasoning, not free association.
Free association would add more confusion. This ordering was discovered in testing.

---

## Part 4: SubstrateSnapshot — Provenance at Crystallization

When a concept crystallizes (gate → FLOW), we capture WHAT the substrates
observed at that moment. This is the `SubstrateSnapshot`:

```rust
pub struct SubstrateSnapshot {
    birth_substrate: Substrate,           // which was primary
    birth_route: SubstrateRoute,          // full routing config
    birth_signals: SubstrateSignals,      // signals at crystallization
    birth_coherence: Coherence,           // were substrates in agreement?

    // Per-substrate state (only if that substrate was active)
    nars_truth: Option<(f32, f32)>,       // (frequency, confidence)
    theta_at_birth: Option<f32>,          // BCM theta average
    maturity_at_birth: Option<u8>,        // structural stability
    semantic_nearest_at_birth: Option<f32>, // cosine to nearest
    saturation_at_birth: Option<f32>,     // soaking completeness
}
```

The snapshot is constructed via `SubstrateSnapshot::capture()` which examines
the route to determine which per-substrate fields to populate.

### Recall Route Derivation

The snapshot produces a `recall_route()` — suggesting which substrates to
activate when this atom is later recalled:

```
Convergent birth → same route for recall (it worked, do it again)
Divergent birth  → add Evidential if not already present (disagreement
                   existed at birth — re-check on recall)
Partial/Singular → same route for recall
```

This means: **concepts born in disagreement carry a "re-check" flag.**
When you recall "love" and it was crystallized during substrate disagreement,
the recall will naturally activate more substrates to verify.

---

## Part 5: Wiring Into CollapseGate

The existing CollapseGate (layer_stack.rs) operates on binary conflict detection:
AND + popcount between delta layers. The substrate signals add a NEW channel:

```
CURRENT GATE DECISION:
  conflict = popcount(delta[i] AND delta[j])
  if conflict / total_bits > threshold → BLOCK
  if conflict / total_bits < threshold → FLOW
  else → HOLD

NEW GATE DECISION (augmented):
  binary_conflict = popcount(delta[i] AND delta[j])  — existing
  coherence = coherence(substrate_signals)             — NEW

  if coherence == Convergent:
    → lower threshold for FLOW (substrates agree, trust the binary signal)
  if coherence == Divergent:
    → raise threshold for FLOW (substrates disagree, accumulate more evidence)
    → HOLD is the safe default
  if coherence == Singular:
    → use existing threshold (single substrate, no cross-check available)
```

The coherence MODULATES the gate threshold — it doesn't replace the binary
conflict detection. Binary conflict is the structural signal (hot path, 2ns).
Coherence is the multi-modal signal (warm path, computed from aggregated
substrate signals after the binary path completes).

```
Binary conflict = fast, coarse, structural
Coherence       = slow, nuanced, semantic
```

The gate uses BOTH. Binary conflict eliminates obvious contradictions.
Coherence determines whether the remaining superposition is trustworthy.

---

## Part 6: The Cognitive Cycle (Revised from Prompt 20)

```
INPUT: text arrives
  ↓
CURRENT STYLE: determines SubstrateRoute (crewai-rust)
  ↓
SUBSTRATE DISPATCH: route.primary does main work
  ├─ STRUCTURAL: Hamming search → hits, distances
  ├─ SOAKING: organic_deposit → saturation, θ, maturity
  ├─ EVIDENTIAL: NARS revision → confidence, contradictions
  └─ SEMANTIC: Jina cosine → nearest, neighborhood
  ↓
SIGNAL COLLECTION: SubstrateSignals aggregated
  ↓
COHERENCE: coherence(signals) → Convergent/Partial/Divergent
  ↓
BINARY CONFLICT: delta AND popcount (existing gate logic)
  ↓
AUGMENTED GATE: CollapseGate with coherence-modulated threshold
  ├─ FLOW:  SubstrateSnapshot::capture() → store atom
  │         ThinkingAtom = binary planes + snapshot provenance
  ├─ HOLD:  keep superposition, keep soaking, no transition
  └─ BLOCK: recommend_transition() → suggest substrate change
            If transition recommended → new SubstrateRoute → different processing
  ↓
STYLE TRANSITION (crewai-rust):
  Substrate recommendation → ThinkingStyle transition
  (Structural→Soaking = "start learning", Soaking→Structural = "crystallize")
  ↓
REPEAT
```

---

## Part 7: The Dependency Direction (LAW)

```
rustynum-core (types + SIMD + substrate.rs)
    ↑
    │ provides: Substrate, SubstrateRoute, SubstrateSignals,
    │           Coherence, coherence(), recommend_transition(),
    │           SubstrateSnapshot, soaking_signals()
    │
rustynum-holo (holographic containers)
    ↑
ladybug-rs (BindSpace + CollapseGate integration)
    ↑
    │ wires: SubstrateSignals into CollapseGate threshold modulation
    │ uses: SubstrateSnapshot for atom provenance
    │
crewai-rust (ThinkingStyle + cortex dispatch)
    ↑
    │ owns: ThinkingStyle → SubstrateRoute mapping
    │ owns: ThinkingAtom = Fingerprint planes + SubstrateSnapshot + style + merkle
    │ calls: recommend_transition() → substrate suggestion
    │ maps:  substrate suggestion → ThinkingStyle transition
```

rustynum provides the compute primitives (Substrate enum, coherence function,
transition recommender, snapshot capture). crewai-rust provides the cognitive
glue (ThinkingStyle mapping, cortex dispatch, style transitions).

This respects the architectural law: rustynum never imports BindSpace, crewai-rust,
n8n-rs, or neo4j-rs. It is a compute + type leaf with zero IO.

---

## Part 8: Connection to Existing Literature

### Multi-Sensory Integration (Ernst & Banks 2002)

Our inverse-variance weighting is the standard model for how the brain
combines visual and haptic cues. The key insight: optimal combination
weights each modality by the RECIPROCAL of its variance.

Our substrates map to sensory modalities:
- Structural = "vision" (high acuity, binary, fast)
- Soaking = "proprioception" (slow, continuous, self-referenced)
- Evidential = "auditory" (sequential, accumulating, temporal)
- Semantic = "haptic" (coarse, broad, spatial)

### Cue Conflict (Landy, Maloney, Johnston, Young 1995)

When cues disagree beyond threshold, the brain doesn't average — it
SELECTS the more reliable modality. Our Coherence::Divergent triggers
the same response: don't average, investigate.

### BCM Metaplasticity (Bienenstock, Cooper, Munro 1982)

The θ sliding threshold is already implemented in organic.rs.
SubstrateSignals.theta_average carries this into the coherence computation:
high θ = well-trained = low variance = high weight in combination.

### NARS Evidential Reasoning (Wang, Hammer, et al.)

NARS (f,c) truth values directly populate SubstrateSignals.nars_confidence
and nars_frequency. The evidence count k determines how much the evidential
substrate's "vote" should weigh in coherence computation.

---

## Part 9: What This Session Added

### New File: `rustynum-core/src/substrate.rs`

| Item | Type | Purpose |
|------|------|---------|
| `Substrate` | enum | The four computational substrates |
| `SubstrateRoute` | struct | Routing config (primary + optional secondary/tertiary) |
| `SubstrateSignals` | struct | Aggregated signals from all active substrates |
| `Coherence` | enum | Cross-substrate agreement classification |
| `SubstrateSnapshot` | struct | Provenance captured at crystallization |
| `coherence()` | fn | Pairwise agreement → Coherence |
| `recommend_transition()` | fn | Signals → substrate change suggestion |
| `soaking_signals()` | fn | SynapseState register → (saturation, θ_avg, maturity_avg) |
| `fill_soaking_signals()` | fn | Populate soaking fields of SubstrateSignals |

### Test Counts

| Module | Tests | Status |
|--------|-------|--------|
| substrate.rs | 24 | All passing |
| organic.rs | 21 | All passing |
| soaking.rs | 14 | All passing |
| three_plane.rs | ~20 | All passing |
| rustynum-core total | 399 | All passing |

### What Does NOT Change

- `layer_stack.rs` — CollapseGate stays pure binary conflict (augmentation wires upstream)
- `organic.rs` — BCM plasticity unchanged
- `soaking.rs` — int8 bridge unchanged
- `three_plane.rs` — Arrow schemas unchanged
- All prompts 01–20

---

## Wiring Checklist (for NEXT ice cakes, not this one)

```
□ ladybug-rs: modulate CollapseGate threshold by Coherence
□ crewai-rust: implement ThinkingStyle → SubstrateRoute mapping (using SubstrateRoute constructors)
□ crewai-rust: implement ThinkingAtom = Fingerprint planes + SubstrateSnapshot
□ crewai-rust: implement recommend_transition() → ThinkingStyle transition mapping
□ crewai-rust: cortex.rs dispatch loop uses SubstrateRoute
□ ladybug-rs: SubstrateSnapshot stored in Lance alongside binary planes
□ ladybug-rs: recall_route() used when loading atoms from storage
□ rustynum-arrow: add SubstrateSnapshot serialization to bind_nodes_v2 schema
```

---

## Summary

The four substrates are not independent — they form a coherence field.
Convergence means the system is confident. Divergence means the system
needs to investigate. The CollapseGate uses coherence to MODULATE its
threshold: convergent → lower barrier to crystallize, divergent → raise
barrier, accumulate more evidence.

The SubstrateSnapshot captures this coherence at birth, and the recall_route
carries it forward: concepts born in agreement are recalled simply;
concepts born in disagreement are recalled with extra verification.

This is the minimum viable multi-sensory integration:
- Not a vote (majority doesn't rule)
- Not an average (that destroys the conflict signal)
- Inverse-variance weighting + conflict detection
- The conflict IS the awareness signal
- Same principle as BCM, same principle as the brain

```
Signal → Weight → Combine → Conflict? → Investigate : Commit
  ↑                                         ↑
  └── substrate-specific ──── cross-substrate ──→ gate decision
```

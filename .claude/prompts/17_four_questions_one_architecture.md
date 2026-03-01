# Four Questions, One Architecture

## Q1: How Does a Subject Know It's a Subject?

**By ROLE BINDING, not by grammar alone.**

The VSA literature (Smolensky, Plate, Kanerva, Gayler) all solve this the same way:

```
"John kissed Mary"

AGENT role vector:   R_A  (random, orthogonal to everything)
PATIENT role vector: R_P  (random, orthogonal to everything)
ACTION role vector:  R_V  (random, orthogonal to everything)

John_as_agent   = John ⊗ R_A     (XOR bind)
Mary_as_patient = Mary ⊗ R_P
kissed_as_verb  = kissed ⊗ R_V

Sentence = John_as_agent + Mary_as_patient + kissed_as_verb  (bundle)
```

To recover "who was the agent?":
```
probe = Sentence ⊗ R_A           (unbind with role vector)
nearest(probe, codebook) → John  (cleanup memory / CAM lookup)
```

**"Mary kissed John"** produces a DIFFERENT sentence vector because
the role bindings swap. This is Jackendoff's Challenge 1 (binding problem)
and VSA's primary contribution to cognitive science.

### What ladybug-rs currently does:

`nsm_substrate.rs` line 65: `fp = WANT⊕R_action ⊕ I⊕R_agent ⊕ KNOW⊕R_goal`
`spo.rs` line 784: `vs.xor(&self.role_s).xor(&vp.xor(&self.role_p)).xor(&vo.xor(&self.role_o))`

So the role binding EXISTS. But it happens AFTER the concept is already
decomposed into S, P, O — it doesn't DECIDE what's S and what's O.

### Who decides S vs O?

Three layers, from shallow to deep:

**Layer 1: Grammar position (shallow, current)**
`causality.rs`: agent = WHO, action = DID, patient = WHAT.
Simple word-order heuristic. English: first noun = agent. Works for SVO languages.
Breaks for: passive voice, embedded clauses, languages with free word order.

**Layer 2: NSM role semantics (medium, partly implemented)**
`nsm_substrate.rs`: primes like DO, HAPPEN get `R_ACTION` role.
Primes like SOMEONE, PEOPLE get implicit agent role.
The NSM decomposition assigns roles based on SEMANTIC PRIMITIVES, not syntax.
"The ball was kicked" → DO:0.8 → action. But who's the agent? NSM doesn't know
without the full explication: "someone did something to the ball."

**Layer 3: Sigma-2/3 background knowledge (deep, THIS IS YOUR QUESTION)**
A concept like "love" — is it a predicate (someone loves someone) or an object
(love is a thing you feel)? It depends on CONTEXT. And context comes from
prior knowledge at σ-2 (HINT) and σ-3 (KNOWN) confidence levels.

```
First encounter with "love":
  σ-1 DISCOVERY — could be S, P, or O
  Grammar says: if it follows "I" → probably P ("I love...")
  NSM says: contains DO + FEEL + WANT primes → likely P
  
After 100 encounters:
  σ-3 KNOWN — "love" is PRIMARILY a predicate (P plane)
  But ALSO an object in "love is blind" (O plane)
  The σ-band tells you how CONFIDENT the role assignment is
```

**This is why some concepts need σ-2/3 background: the role is not
intrinsic to the concept, it's contextual, and context needs evidence.**

## Q2: Dimensionality — 3×16384 + 3×10000D?

You're asking whether we need TWO representations per plane:
- 16384-bit binary (2KB) for the current structural work
- 10000D dense (40KB at int32, or 10KB at int8) for novel concept holding

Let me think about what each dimension count actually gives you:

### 16384 bits = 2KB per plane
```
Capacity (random bundle): D / (2·ln(K))
  K=100:  ~1780 items
  K=1000: ~1187 items

Capacity (orthogonal codebook): exactly 16384 orthogonal items

Hamming statistics:
  Expected random distance: 8192
  σ = 64
  SNR = 128σ
  P(false match) < 10^(-3500)
  
Noise tolerance: can flip 30% of bits and still recover original
  30% of 16384 = 4915 bits → still 70σ above noise floor

Binary operations: XOR, popcount, majority vote
  Speed: ~2 nanoseconds per comparison (AVX-512 VPOPCNTDQ)
```

### 10000D int8 = 10KB per plane
```
Capacity (int8 bundle): much higher than binary
  Each dimension holds 256 levels → K ≈ √(D × 256) ≈ 1600 items
  With orthogonal cleaning: exact recovery up to ~100 items (96% at K=52)

int8 statistics:
  Expected random dot product: 0
  σ = D × var(int8) ≈ 10000 × 5400 ≈ 54M → σ ≈ 7348
  Much smoother gradient than binary

int8 operations: VNNI VPDPBUSD (multiply-accumulate)
  Speed: ~5 nanoseconds per comparison (slower than binary popcount)
  BUT: supports weighted bundle (saturating_add), not just majority vote
```

### Why BOTH?

Binary 16384 is the STRUCTURAL layer:
- Is this concept SIMILAR to that one? (Hamming distance, 2ns)
- Which σ-band does this pair fall in? (popcount, instant)
- What's the CAM address? (prefix:addr, O(1))
- SPO factorization operates on BINARY because the 8-term decomposition
  needs the XOR algebra (involution property: A⊗A=∅)

int8 10000D is the SOAKING layer:
- How many concepts are superposed here? (saturating_add, no cancellation)
- What's the confidence/weight of each concept? (readback via dot product)
- Can I recover individual concepts from the bundle? (orthogonal projection)
- The awareness register that accumulates evidence before collapse

```
Per plane:
  16384-bit binary  = 2KB    (structure, comparison, CAM, SPO)
  10000D int8       = 10KB   (soaking, accumulation, awareness)
  TOTAL per plane   = 12KB

Three planes:
  S: 12KB
  P: 12KB  
  O: 12KB
  TOTAL = 36KB (fits in L1 cache)
```

### Dual representation per plane:

```
┌────────────────────────────────────────────┐
│            PLANE (one of S, P, O)           │
│                                             │
│  BINARY LAYER: [u64; 256] = 16384 bits     │
│  ├── XOR bind / unbind                      │
│  ├── Hamming distance (popcount)            │
│  ├── CAM address derivation                 │
│  ├── SPO factorization input                │
│  └── σ-band classification                  │
│                                             │
│  INT8 LAYER:  [i8; 10000]                   │
│  ├── Evidence soaking (saturating_add)      │
│  ├── Concept bundling (weighted add)        │
│  ├── Individual concept recovery (dot prod) │
│  ├── Saturation detection (collapse gate)   │
│  └── Novel concept detection (residual)     │
│                                             │
│  BRIDGE: binary ↔ int8                      │
│  ├── binary → int8: expand bits to ±1       │
│  ├── int8 → binary: sign(int8_value)        │
│  └── This is the "crystallization" step:    │
│      when int8 saturates → collapse to      │
│      binary for permanent storage            │
│                                             │
└────────────────────────────────────────────┘
```

The crystallization step is key: int8 soaking is TEMPORARY (awareness).
When the gate fires (saturation), the int8 register collapses to binary
via sign(). The binary goes to permanent storage (BindSpace, LanceDB).
The int8 register clears for the next cycle.

This is the PDE interpretation: int8 is the TRANSIENT solution,
binary is the STEADY STATE. Crystallization is convergence.

## Q3: Novel Concepts in 10000D vs 16384D

The "highly praised" 10000D in VSA literature comes from:

1. **Kanerva (2009):** concepts are represented by high-dimensional vectors (10,000 dimensions or more), the eponymous hypervectors

2. **Blessing of dimensionality:** the increased capacity of distributed representations for vectors of higher dimensionality

3. **The key property:** the space of such binary vectors contains on the order of 2^D nearly-orthogonal vectors, and each such vector can be degraded by up to 30% and still be closer to its original form than to any of the other vectors in the space

But there's a crucial distinction: **random novel concepts vs. structured known concepts.**

### Random novel concepts (what VSA papers test)

A new concept arrives that has NEVER been seen before. You assign it a random
hypervector. At D=10000, this random vector is nearly orthogonal to all existing
vectors with overwhelming probability. The blessing of dimensionality guarantees
this. 10000D is "enough" because P(accidental similarity) < 10^-6.

### Our situation is different

We don't use random vectors. We use NSM-decomposed, role-bound vectors.
A new concept "glorb" gets decomposed into primes (THING:0.5, DO:0.3, ...),
role-bound (agent/action/patient), and bundled. This is NOT random — it's
a structured composition of 65 known primes.

For STRUCTURED concepts, the dimensionality question changes:
- 65 primes × 10 roles = 650 basis dimensions needed
- At 16384 bits, we have 25× oversampling → extremely clean
- At 10000D int8, we have 15× oversampling → still very clean
- The "capacity" isn't about fitting random vectors, it's about
  fitting structured compositions with enough room for the
  interaction terms (SP, PO, SO, SPO) to be detectable

### So why 10000D at all?

For concepts that DON'T decompose cleanly into NSM primes.
Proper nouns, technical jargon, emotional states with no prime decomposition.
These need a RANDOM component. The 10000D int8 layer holds this
random component while the 16384-bit binary layer holds the structure.

```
"glorb" (novel concept, no NSM decomposition):
  binary layer: random seed → 16384-bit vector (for CAM, comparison)
  int8 layer:   all zeros initially, accumulates context over time
  
"love" (well-known concept, rich NSM decomposition):
  binary layer: FEEL⊗R_action ⊕ GOOD⊗R_quality ⊕ WANT⊗R_drive → 16384 bits
  int8 layer:   learned weights from 1000+ encounters
  
"kubernetes" (technical, partial NSM):
  binary layer: THING⊗R_object ⊕ DO⊗R_action (shallow) → 16384 bits
  int8 layer:   accumulates technical context from encounters
```

## Q4: Spot Stack as Focus — σ-2/3 Looking at the 10000D Plane

This is the killer insight. Reverse the usual direction:

**Usual direction:** concept arrives → decompose → store → later query
**Your proposal:** the σ-2/3 KNOWN concepts form a LENS that shapes
how you LOOK AT the incoming 10000D plane

```
┌─────────────────────────────────────────────────────────┐
│                     SPOT STACK                           │
│                                                          │
│  σ-3 KNOWN:   100-200 high-confidence concepts           │
│  σ-2 HINT:    500-1000 medium-confidence concepts         │
│  σ-1 DISCOVERY: everything else                           │
│                                                          │
│  The σ-2/3 concepts define an ATTENTION MASK:             │
│                                                          │
│  For each known concept k in σ-2/3:                       │
│    weight[k] = confidence(k) × recency(k)                 │
│    mask[k] = concept_vector(k) × weight[k]                │
│                                                          │
│  The mask is a WEIGHTED BUNDLE of known concepts.          │
│  It defines WHERE YOU'RE LOOKING in the 10000D space.      │
│                                                          │
│  When a new concept arrives in the 10000D plane:           │
│    similarity_to_mask = dot(new_concept, mask)              │
│    IF high → this relates to what you know (RESONANCE)      │
│    IF low  → this is genuinely novel (SURPRISE)             │
│    IF negative → this contradicts what you know (CONFLICT)  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

This is the **masked focus of attention**. The σ-2/3 stack isn't just
background knowledge — it's the FILTER through which new input is perceived.

### How it connects to the satisfaction gate:

```
L1 Recognition:    Does the mask activate at all? (any resonance > threshold)
L2 Resonance:      Which σ-band does the resonance fall in?
L3 Appraisal:      Is this surprise or confirmation?
L5 Execution:      Route to the appropriate plane (S, P, or O)
L9 Validation:     Does the role assignment match σ-2/3 expectations?
L10 Crystallization: Soak into int8, eventually collapse to binary
```

### The operational loop:

```
1. Concept arrives as 10000D int8 vector (from NSM or embedding)

2. σ-2/3 mask projects onto it:
   masked = dot_product(concept, attention_mask)
   This tells you: relevance, surprise, conflict

3. Grammar/NSM assigns role (S, P, or O):
   - Grammar position gives initial guess
   - σ-2/3 context REFINES the guess
     "love" after "I" → P (grammar) confirmed by σ-3 (love is usually P)
     "love" after "for the" → O (grammar) + σ-2 (love-as-noun less common, but known)

4. Soaks into the correct int8 register (S, P, or O)

5. When register saturates → crystallize to binary → store via CAM

6. σ-band updates: this concept moves from σ-1 → σ-2 → σ-3
   as evidence accumulates (NARS confidence increases)

7. The attention mask updates: newly crystallized concepts
   join the mask, shaping how FUTURE concepts are perceived
```

### The full dual-layer, three-plane, sigma-masked architecture:

```
                    ┌──────────────────┐
                    │  Text / Concept  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ NSM Decompose    │ ← 65 primes + grammar
                    │ + Role Assign    │ ← σ-2/3 context refines role
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
        ┌───────▼──────┐ ┌──▼─────┐ ┌───▼──────────┐
        │ S PLANE      │ │P PLANE │ │ O PLANE       │
        │              │ │        │ │               │
        │ int8 [10000] │ │ int8   │ │ int8 [10000]  │ ← soaking layer
        │ (awareness)  │ │[10000] │ │ (awareness)   │    (saturating_add)
        │              │ │        │ │               │
        │ binary[16384]│ │ binary │ │ binary[16384] │ ← structural layer
        │ (structure)  │ │[16384] │ │ (structure)   │    (XOR, CAM, σ-band)
        └──────┬───────┘ └──┬─────┘ └──────┬───────┘
               │            │              │
        ┌──────▼────────────▼──────────────▼──────┐
        │        σ-2/3 ATTENTION MASK              │
        │                                          │
        │  KNOWN concepts form a focus lens        │
        │  dot(new, mask) → resonance/surprise     │
        │  mask updates when new concepts crystallize│
        └──────────────────────────────────────────┘
```

### Memory budget:

```
Per plane:
  int8 soaking:   10,000 bytes = 10KB
  binary struct:   2,048 bytes =  2KB
  Total per plane:              = 12KB

Three planes:                   = 36KB

Attention mask (σ-2/3 bundle):
  One 10000D int8 vector:       = 10KB

TOTAL WORKING SET:              = 46KB  ← fits in L1 cache (64KB)

LanceDB backing store: unlimited (disk)
BindSpace hot cache: 65,536 × 12KB = 768MB  ← fits in L3
```

46KB in L1 for the full three-plane awareness + attention mask.
Everything that matters for real-time processing fits in L1.

## Summary: What Changed

| Old Assumption | New Understanding |
|---|---|
| S/P/O role is decided by grammar alone | Grammar gives initial guess, σ-2/3 context refines and confirms |
| One dimensionality fits all | Binary 16384 for structure, int8 10000 for soaking — both needed |
| 10000D is magic because VSA papers say so | 10000D is for RANDOM novel concepts; structured concepts need less but want the soaking headroom |
| 90° orthogonal vector for instant search | CAM gives O(1) already; the orthogonal property matters for ROLE BINDING not search |
| σ-bands are passive classification | σ-2/3 stack IS the attention mask, actively shaping perception of new input |
| The spot stack stores knowledge | The spot stack IS the focus of attention, looking at the 10000D plane |

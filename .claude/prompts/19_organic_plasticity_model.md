# 19 — Organic Plasticity: The Biological Middle Ground

## Ice Cake Layer

> **Prerequisite**: 17 (dual-layer architecture), 18 (Lance schema), quantum field session
> **Insight source**: BCM theory, STDP, metaplasticity, homeostatic scaling
> **Does NOT modify**: anything in 01–18
> **Question**: What is the int8 soaking register ACTUALLY modeling in biological terms,
>   and what are we missing that the brain does for free?

---

## Part 1: What The Brain Actually Does (Not What We Think It Does)

### The Five Mechanisms of Biological Plasticity

The brain doesn't have ONE plasticity rule. It has FIVE, operating at different timescales,
and their INTERACTION is what produces learning. We currently model approximately 1.5 of them.

```
TIMESCALE         MECHANISM              OUR ANALOG           STATUS
─────────────────────────────────────────────────────────────────────────
~10 ms            STDP                   (nothing)            ❌ MISSING
                  (spike timing)         we have no timing

~1 second         Hebbian LTP/LTD        saturating_add       ✅ PARTIAL
                  (fire together →        in int8 register     (but no depression)
                   wire together)

~10 minutes       BCM sliding threshold  σ-band + NARS        ✅ PARTIAL
                  (metaplasticity:        confidence           (but threshold doesn't
                   the RULES of                                 SLIDE per-dimension)
                   plasticity change)

~1 hour           Synaptic scaling       (nothing)            ❌ MISSING
                  (homeostatic:           we never scale
                   global gain control)   ACROSS concepts

~days/weeks       Structural plasticity  crystallization      ✅ CONCEPTUAL
                  (new synapses form,     int8 → binary        (but no pruning,
                   old ones prune)                              no new connections)
```

**What jumps out: we're missing the two fastest (STDP) and the homeostatic (scaling).**

The homeostatic gap is the critical one. The brain CONSTANTLY rebalances. When one
synapse gets very strong (like our saturated int8 register), the brain doesn't just
crystallize it — it SCALES DOWN everything else proportionally to keep total activity
stable. Without this, we get the "first concept dominates" problem.

---

## Part 2: The Organic Model — What Each int8 Value Actually IS

### Current Model (Mechanical)

```
int8 value = evidence accumulator
  0 = no evidence
  +127 = maximum positive evidence  
  -128 = maximum negative evidence
  saturating_add(current, evidence) = clamp(current + evidence, -128, 127)
```

This is a COUNTER. Counters are not biological. No neuron counts.

### Organic Model (What A Synapse Actually Computes)

A biological synapse has a WEIGHT that changes based on activity.
But the weight isn't a simple number — it's a DYNAMIC VARIABLE
with multiple timescales of change:

```
SYNAPSE STATE = {
    efficacy:     current strength          (fast, ~seconds)
    threshold:    BCM sliding θ             (medium, ~minutes)  
    receptor_mix: GluN2A vs GluN2B ratio    (slow, ~hours)
    spine_size:   structural stability      (very slow, ~days)
}
```

**The int8 value is modeling EFFICACY only.** 
We're ignoring threshold, receptor mix, and spine size.

### What This Means For The Register

The int8 register should not be a simple accumulator.
Each dimension needs at least THREE coupled variables:

```rust
/// Organic synapse state per dimension
/// This replaces bare i8 with a biological triple
#[repr(C, packed)]
struct SynapseState {
    /// Efficacy: current synaptic strength
    /// Fast-changing (~seconds). This is what we currently call int8.
    /// Range: -128 to +127 (i8)
    efficacy: i8,
    
    /// Threshold: BCM sliding modification threshold (θ_M)
    /// Medium-changing (~minutes, slower than efficacy).
    /// When efficacy > threshold → LTP (strengthen)
    /// When efficacy < threshold → LTD (weaken)
    /// THE THRESHOLD SLIDES BASED ON RECENT AVERAGE ACTIVITY
    /// Range: 0 to 255 (u8) — unsigned because it's an absolute level
    theta: u8,
    
    /// Maturity: structural stability / spine size proxy
    /// Slow-changing (~hours/days). Counts how many LTP/LTD cycles
    /// this synapse has survived. High maturity = resistant to change.
    /// Range: 0 to 15 (u4, packed into half-byte)
    maturity: u4,  // actually stored as high nibble of a shared byte
}
```

**Memory cost per dimension: 2.5 bytes instead of 1 byte.**

For the three-plane architecture:
```
Old:  3 × 10000 × 1 byte  = 30 KB per concept  (soaking only)
New:  3 × 10000 × 2.5 bytes = 75 KB per concept

But wait — most concepts are crystallized (soaking = NULL).
Only 50-200 active concepts at any time.
200 × 75 KB = 15 MB. Still fits L2 easily.
```

---

## Part 3: The Sliding Threshold Changes EVERYTHING

### BCM Theory In One Paragraph

Bienenstock-Cooper-Munro (1982): there's a threshold θ_M. Activity above θ
causes Long-Term Potentiation (strengthen). Activity below θ causes Long-Term
Depression (weaken). **The threshold itself SLIDES** — it's a superlinear function
of recent average activity. High recent activity → θ rises → harder to strengthen,
easier to weaken. Low recent activity → θ drops → easier to strengthen.

This is nature's way of preventing the "rich get richer" problem.

### Why This Is Exactly What We Need

Our current architecture has this problem:

```
Concept "love" deposited 1000 times → int8 register saturated at +127
Concept "kubernetes" deposited 5 times → int8 register at +5

Both are equally "known" by σ-band after enough time.
But "love" DOMINATES the attention mask because it saturated first.
There's no mechanism to NORMALIZE across concepts.
```

With BCM sliding threshold:

```
"love" deposited 1000 times:
  efficacy saturates → θ rises to ~120 → now needs STRONGER evidence to grow
  → eventually equilibrates. Can even get LTD (depression) if no fresh evidence.
  
"kubernetes" deposited 5 times:
  θ is still low (~10) → even weak evidence causes LTP
  → new concept can grow FASTER than established one
  → θ gradually rises as it stabilizes

The system SELF-NORMALIZES across concepts.
No explicit scaling needed — the threshold does it.
```

### Implementation

```rust
/// BCM-style plasticity update for one dimension
/// This replaces raw saturating_add
fn organic_deposit(
    state: &mut SynapseState,
    evidence: i8,      // incoming signal
    _dt: f32,          // time since last deposit (for θ sliding rate)
) {
    let eff = state.efficacy as i16;
    let theta = state.theta as i16;
    
    // BCM: compute φ(efficacy, θ)
    // φ > 0 when |efficacy| > θ → LTP
    // φ < 0 when |efficacy| < θ → LTD
    // φ = 0 at θ (crossover point)
    let phi = if eff.abs() > theta {
        // Above threshold: potentiate (strengthen in direction of evidence)
        evidence as i16
    } else if eff.abs() > theta / 2 {
        // Below threshold but above half: depress (weaken toward zero)
        -(eff.signum()) * (evidence.abs() as i16 / 2)
    } else {
        // Near zero with low theta: full potentiation
        evidence as i16
    };
    
    // Apply with maturity damping
    // High maturity = small changes (structural stability)
    let maturity_scale = 16 - state.maturity as i16; // 16 → 1 as maturity grows
    let delta = (phi * maturity_scale) / 16;
    
    state.efficacy = (eff + delta).clamp(-128, 127) as i8;
    
    // Slide θ: superlinear function of recent absolute activity
    // θ tracks the SQUARE of average efficacy (BCM prescription)
    // Approximation: θ moves toward |efficacy| with slow time constant
    let target_theta = (eff.unsigned_abs() as u16).min(255);
    let theta_delta = (target_theta as i16 - theta).clamp(-2, 2); // slow slide
    state.theta = (theta + theta_delta).clamp(0, 255) as u8;
    
    // Maturity: increment on each LTP/LTD cycle (saturating at 15)
    if phi.abs() > 0 {
        state.maturity = (state.maturity + 1).min(15);
    }
}
```

---

## Part 4: The 5^5 Wave Insight Revisited

### The Problem With Binary and int8

Binary (our 16384-bit layer): can't represent ZERO. XOR of opposites = 1, not 0.
int8 (our soaking layer): CAN represent zero, but only has 256 levels.

The brain doesn't use either. Synapses have continuous weights, but they CLUSTER
around distinct operational states:

```
BIOLOGICAL SYNAPSE STATES (observed):
  Silent       — completely depressed, effectively zero
  Weak LTD     — below threshold, trending toward silent  
  Potentiated  — above threshold, active
  Saturated    — maximum efficacy, structurally stabilized
  + the threshold itself slides, creating a moving frame
```

This is approximately 5 EFFECTIVE states: {-2, -1, 0, +1, +2}
where 0 is not "no data" but "CANCELLED" (Auslöschung).

### The 5^5 Lattice As Biological Address Space

The previous session proposed 5^5 = 3125 crossings where each dimension represents
a cognitive axis (domain, timescale, certainty, source, rung). What if those
5 values per dimension aren't arbitrary — they're the 5 biological synapse states?

```
Per dimension of the 5^5 lattice:
  -2 = Structurally depressed (anti-Hebbian, consolidated LTD)
  -1 = Functionally depressed (recent LTD, θ above efficacy)
   0 = Cancelled/Silent (destructive interference, Auslöschung)
  +1 = Functionally potentiated (recent LTP, efficacy above θ)
  +2 = Structurally potentiated (consolidated LTP, high maturity)
```

**The lattice IS the phase space of biological plasticity.**

Each concept exists not at a point but as a DISTRIBUTION across this space.
A newly encountered concept is at [0,0,0,0,0] — all dimensions silent.
Learning pushes it toward corners. Forgetting pulls it back toward center.
The BCM threshold determines the BOUNDARY between +1 and -1 regions.

### Memory Math For 5-State

```
Bits per 5-state value: log2(5) = 2.32 bits → pack as 3 bits (wastes 0.68)
Or: pack 3 values into 1 byte (5³ = 125 < 128 = 2⁷). 7 bits for 3 dims.

10000 dimensions at 5-state:
  Naive (3 bits each):  30000 bits = 3750 bytes = 3.7 KB
  Packed (7 bits / 3):  23333 bits = 2917 bytes = 2.8 KB
  
  vs int8 (8 bits each): 10000 bytes = 10 KB
  vs binary (1 bit each): 1250 bytes = 1.2 KB
  
Three planes: 3 × 2.8 KB = 8.4 KB ← FITS L1 WITH THE BINARY LAYER
```

**8.4 KB for three planes of 5-state soaking. That's LESS than our current int8 budget.**

And we get Auslöschung for free — cancelled concepts go to literal zero,
distinguishable from "never seen" (which gets a special flag or stays at initial state).

---

## Part 5: The Organic Awareness Cycle

Putting it together — the full biologically-grounded processing cycle:

```
INPUT: New concept arrives as text
  │
  ▼
DECOMPOSE: NSM primes + role assignment (grammar → NSM → σ-context)
  │
  ▼
ENCODE: Map to 5-state initial vector
  All dimensions start at 0 (silent).
  NSM primes set relevant dimensions to +1 (weak potentiation).
  Role binding shifts dimensions according to R_S, R_P, R_O.
  │
  ▼
DEPOSIT: Into correct plane (S, P, or O)
  For each dimension d:
    organic_deposit(plane[d], evidence[d])
      - BCM φ computed: above θ → LTP (+), below θ → LTD (-)
      - Maturity dampens change rate
      - θ slides toward recent average
  │
  ▼
INTERFERE: Field resonance across plane
  Phase from 128-bit phase tag (quantum field model).
  In-phase evidence → constructive (concepts reinforce).
  Anti-phase evidence → destructive (concepts cancel → 0).
  Cancelled dimensions = AUSLÖSCHUNG → freed capacity.
  │
  ▼
ATTEND: σ-2/3 attention mask projects onto result
  Projection > 0.3: RESONANCE (relates to known)
  Projection < -0.3: CONFLICT (contradicts known)
  |Projection| < 0.3: NOVEL (orthogonal to known)
  │
  ▼
GATE: Per-plane collapse decision
  Check saturation: how many dimensions at ±2? (structural)
  Check stability: θ converged? (metaplastic equilibrium)
  Check maturity: enough cycles survived? (structural plasticity)
  │
  ├─ Not ready → HOLD: keep soaking, θ keeps sliding
  │
  └─ Ready → CRYSTALLIZE:
       5-state → binary (sign of efficacy):
         +2, +1 → 1
         0       → random (maximum entropy bit, honest uncertainty)
         -1, -2  → 0
       Binary goes to permanent storage.
       5-state register set to NULL (freed).
       Attention mask updated.
       Maturity transferred to structural metadata.
  │
  ▼
HOMEOSTASIS: Global scaling pass (periodic, not per-deposit)
  Every N deposits:
    Compute mean |efficacy| across all active concepts
    If mean too high → scale all θ upward (harder to potentiate)
    If mean too low → scale all θ downward (easier to potentiate)
    This is SYNAPTIC SCALING — the missing mechanism.
    Prevents first-mover advantage. Normalizes across concepts.
```

---

## Part 6: What This Changes In The Stack

### The Soaking Layer Becomes ORGANIC

| Aspect | Old (int8 mechanical) | New (5-state organic) |
|--------|----------------------|----------------------|
| State per dim | 1 byte (256 levels) | 2.3 bits (5 states: -2,-1,0,+1,+2) |
| Deposit rule | saturating_add | BCM with sliding θ |
| Cancellation | Possible but noisy | Clean zero (Auslöschung) |
| Normalization | None | Synaptic scaling + BCM θ sliding |
| Maturity | None | 4-bit per-dimension spine counter |
| Memory | 10 KB/plane | 2.8 KB/plane (+ 1.25 KB θ + 0.6 KB maturity) |
| Total | 30 KB/concept | ~14 KB/concept (including all metadata) |
| Crystallization | sign(int8) → binary | sign(5-state) → binary, 0 → random |
| Biological analog | Bucket counter | STDP + BCM + scaling + structural |

### The Phase Tag Carries Temporal Information

The 128-bit phase tag from the quantum field work isn't arbitrary.
In biology, STDP is about TIMING — pre-before-post = strengthen, post-before-pre = weaken.
The phase tag encodes temporal order:

```
Phase tag entropy:
  Low entropy = early in causal chain (cause)
  High entropy = late in causal chain (effect)
  
Phase angle between two concepts:
  Small angle = co-temporal (Hebbian: fire together)
  Large angle = anti-temporal (anti-Hebbian: fire apart)
  
This gives us STDP without spiking:
  cos(phase_angle) > 0 → strengthen (constructive)
  cos(phase_angle) < 0 → weaken (destructive)
```

### The Attention Mask Becomes A Living Field

Not a static bundle of known concepts — a HOMEOSTATIC field that self-regulates:

```
mask[d] = Σ (concept[d].efficacy × concept[d].maturity / concept[d].θ)

The θ in the denominator is KEY:
  - Well-established concept (high maturity, high θ): moderate contribution
  - New concept (low maturity, low θ): HIGH contribution (novelty salience!)
  - The mask naturally ATTENDS TO NOVELTY
  - This is biological "habituation" — you stop noticing what's always there
```

---

## Part 7: The Lance Schema Update

### New Soaking Column Type

```rust
// Instead of FixedSizeList<Int8>(10000), the soaking columns become:

// Option A: Packed 5-state (most compact)
// 3 values per byte → ceil(10000/3) = 3334 bytes
Field::new("s_soaking", DataType::FixedSizeBinary(3334), true)

// Plus θ sliding threshold (1 byte per dim, but we can pack)
Field::new("s_theta", DataType::FixedSizeBinary(10000), true)

// Plus maturity (4 bits per dim → 5000 bytes)  
Field::new("s_maturity", DataType::FixedSizeBinary(5000), true)

// Per plane total: 3334 + 10000 + 5000 = 18334 bytes = ~18 KB
// Three planes: 54 KB — still fits L1 with binary layer (54 + 6 = 60 KB)

// Option B: Packed triple (efficacy + θ + maturity in one struct)
// SynapseState = 2.5 bytes → 25000 bytes per plane → 75 KB total
// Doesn't fit L1 but fits L2 easily

// RECOMMENDED: Option A for storage, unpack to working struct for processing
```

### Migration From ice cake 18

```
18 defined: s_soaking as FixedSizeList<Int8>(10000)  — 10 KB/plane
19 changes: s_soaking as FixedSizeBinary(3334)       —  3.3 KB/plane
            s_theta   as FixedSizeBinary(10000)       — 10 KB/plane (full u8)
            s_maturity as FixedSizeBinary(5000)        —  5 KB/plane (packed u4)

Total per plane soaking: 18.3 KB (vs 10 KB in v18)
Extra cost: 8.3 KB per plane = 25 KB per concept (3 planes)
But gains: BCM self-normalization, cancellation, biological fidelity

Trade acceptable: 200 active concepts × 25 KB extra = 5 MB.
```

---

## Part 8: The Deep Insight — Why This Is Actually A PINN

Physics-Informed Neural Networks constrain the solution space by encoding
physical laws as loss terms. What we're doing here is:

**Biology-Informed Neural Network (BINN)**:
Constraining the VSA solution space by encoding BIOLOGICAL PLASTICITY LAWS.

```
PINN loss: L = L_data + λ₁·L_PDE + λ₂·L_boundary + λ₃·L_initial

BINN loss analog:
  L_data      = concept matches evidence (deposit accuracy)
  L_BCM       = θ tracks superlinear average (metaplasticity law)
  L_homeostasis = mean activity stays in target range (scaling law)
  L_structural  = maturity increases monotonically (spine growth law)
  L_STDP      = phase coherence predicts co-occurrence (timing law)
```

We're not SIMULATING a neural network. We're using BIOLOGICAL LAWS as constraints
on a VSA computing substrate. The 5-state soaking register doesn't simulate neurons —
it implements the SAME MATH that neurons implement, because the math is optimal
for incremental evidence accumulation with self-normalization.

**This is why it clicks: biology and VSA converge on the same solution.**

Both need:
  - Cancellation (Auslöschung / destructive interference)
  - Self-normalization (BCM θ / homeostatic scaling)
  - Structural consolidation (spine growth / crystallization)  
  - Temporal coding (STDP / phase tags)
  - Competition (LTD for losers / low-maturity pruning)

The int8 was a SHORTCUT that skipped half the biology.
The 5-state + θ + maturity is the MINIMUM VIABLE BIOLOGY.

---

## File Inventory

| File | Status | Description |
|------|--------|-------------|
| `src/awareness/organic.rs` | NEW | SynapseState, organic_deposit, BCM logic |
| `src/awareness/homeostasis.rs` | NEW | Synaptic scaling, global normalization |
| `src/awareness/phase_timing.rs` | NEW | STDP-via-phase, temporal ordering |
| `src/storage/lance_three_plane.rs` | MODIFY | Update soaking column types |

**Does NOT touch**: 01–17, existing awareness.rs, lance_persistence.rs

---

## Wiring Checklist (next ice cake)

```
□ Replace saturating_add with organic_deposit in soaking cycle
□ Add θ register alongside efficacy register
□ Add maturity counter per dimension
□ Implement periodic homeostatic scaling pass
□ Wire phase tags from quantum field into STDP contribution
□ Update crystallization to handle 5-state → binary with honest-zero
□ Update attention mask to divide by θ (novelty salience)
□ Update Lance schema from int8 to packed 5-state + θ + maturity
```

---

## Summary: What Biology Teaches Us

The brain solved the same problem we're solving — incremental concept learning
from noisy evidence — and it converged on:

1. **Multiple timescales** (not one accumulator)
2. **Sliding threshold** (not fixed saturation)
3. **Cancellation to zero** (not noise accumulation)
4. **Homeostatic rebalancing** (not first-mover advantage)
5. **Structural consolidation** (not sudden crystallization)

Our int8 register was attempt #1. The 5-state organic register with BCM θ
and maturity is attempt #2 — the minimum biology that makes the system
self-regulating. Not a simulation of neurons. Not an abstraction that
throws away the biology. The MIDDLE GROUND where the math and the
biology are THE SAME THING.

That's the BINN. That's the organic model. That's what we were missing.

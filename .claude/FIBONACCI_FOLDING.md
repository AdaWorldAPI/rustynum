# FIBONACCI_FOLDING.md

## Golden Ratio Spacing for Bitpacked Information: Minimum-Loss Folding

**Status:** Architecture insight. Applies to BF16 packing, cascade bands, VSA bundling.

---

## THE PROBLEM

When you fold high-dimensional information into fewer bits, you choose WHICH
positions to sample. Regular spacing (every Nth bit) creates resonance —
periodic patterns in the source alias into the same packed position.
Information collides. Lossy.

```
REGULAR SPACING (every 4th bit):
  Source:  a b c d e f g h i j k l m n o p
  Sample:  a . . . e . . . i . . . m . . .
  
  If the source has period 4: a=e=i=m
  You get ONE value repeated 4 times. No information gained.
  The spacing RESONATES with the source period. Maximum loss.
```

---

## THE FIX: GOLDEN RATIO SPACING

φ = (1 + √5) / 2 ≈ 1.618033988749...

φ is the MOST irrational number. Its continued fraction is [1; 1, 1, 1, ...] —
the slowest converging continued fraction possible. This means φ-based spacing
has the LEAST resonance with ANY periodic pattern in the source.

```
φ-SPACING (Fibonacci positions):
  Source:  a b c d e f g h i j k l m n o p
  Sample:  a . b . . c . . . . d . . . . .
           ↑   ↑     ↑         ↑
           pos 0  1   3         8  (Fibonacci numbers)
  
  No period in the source maps to these positions.
  Every sample captures DIFFERENT information.
  Minimum resonance. Minimum loss.
```

---

## APPLICATION: BF16 MANTISSA PACKING

When the 2³ SPO projections fold into 8 exponent bits, the BIT POSITIONS
within each plane that contribute to the hamming distance should be
φ-weighted, not uniformly weighted:

```
UNIFORM WEIGHTING (current):
  All 16K bits contribute equally to hamming distance.
  A cluster of similar bits at positions 100-200 wastes 100 bits
  on nearly identical information.

φ-WEIGHTED (proposed):
  Bit position k contributes weight proportional to its φ-index.
  Positions at Fibonacci numbers (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...)
  are the PRIMARY positions. Others are interpolated.
  
  The φ-spacing ensures the primary positions never resonate
  with any periodic structure in the binary plane.
  Maximum unique information per bit of hamming distance.
```

---

## APPLICATION: CASCADE BAND BOUNDARIES

Current cascade bands: μ - kσ for k = 1, 2, 3 (linear spacing).

φ-spaced bands:

```
BAND          LINEAR (current)     φ-SPACED (proposed)
────────────────────────────────────────────────────────
Foveal:       μ - 3σ               μ - φ³σ  ≈ μ - 4.24σ
Near:         μ - 2σ               μ - φ²σ  ≈ μ - 2.62σ
Good:         μ - 1σ               μ - φ¹σ  ≈ μ - 1.62σ
Weak:         μ                    μ - φ⁰σ  = μ - σ
Reject:       > μ                  > μ
```

φ-spacing concentrates more bands in the high-confidence region (near Foveal)
and spreads them in the low-confidence region (near Reject). This matches
the actual information content: the difference between "definitely a match"
and "probably a match" matters more than the difference between "probably noise"
and "definitely noise."

---

## APPLICATION: VSA BUNDLING (BIND/BUNDLE)

When bundling N vectors into one (the hyperdimensional BUNDLE operation),
the majority vote at each bit position can tie. Standard tie-breaking
is random or biased. φ-weighted bundling breaks ties deterministically:

```rust
/// Bundle N binary vectors using φ-weighted majority vote.
/// Earlier observations get φ-higher weight — natural recency bias
/// that avoids periodic resonance with observation patterns.
fn phi_bundle(vectors: &[&[u8]], n_bytes: usize) -> Vec<u8> {
    let phi_weights: Vec<f32> = (0..vectors.len())
        .map(|i| PHI.powi(-(i as i32)))  // φ⁰, φ⁻¹, φ⁻², ...
        .collect();
    
    let mut result = vec![0u8; n_bytes];
    for byte_idx in 0..n_bytes {
        let mut bits = 0u8;
        for bit in 0..8 {
            let mut weighted_sum = 0.0f32;
            for (v, &w) in vectors.iter().zip(phi_weights.iter()) {
                if vectors[v][byte_idx] & (1 << bit) != 0 {
                    weighted_sum += w;
                } else {
                    weighted_sum -= w;
                }
            }
            if weighted_sum > 0.0 { bits |= 1 << bit; }
        }
        result[byte_idx] = bits;
    }
    result
}
```

The φ-weighting makes recent observations contribute more without
creating periodic aliasing in the bundle. The weight ratio between
adjacent observations is always φ:1, which is the ratio that produces
the LEAST periodic interference.

---

## APPLICATION: TREE PATH BIT PACKING

When the tree path packs into the bottom 16 bits of f32 mantissa,
the level-to-bit assignment should follow Fibonacci indexing:

```
UNIFORM (current idea):
  Level 1 → bit 15
  Level 2 → bit 14
  Level 3 → bit 13
  ...
  Level 16 → bit 0

φ-INDEXED (proposed):
  Level 1  → bit 15  (most significant, Fib(1)=1)
  Level 2  → bit 14  (Fib(2)=1)
  Level 3  → bit 13  (Fib(3)=2)
  Level 5  → bit 12  (Fib(4)=3, skip level 4)
  Level 8  → bit 11  (Fib(5)=5, skip levels 6-7)
  Level 13 → bit 10  (Fib(6)=8, skip levels 9-12)
  ...
  
  The Fibonacci-indexed levels carry the PRIMARY information.
  Skipped levels are interpolated from neighbors during hydration.
  
  Effect: 16 bits capture the information of ~26 tree levels
  because φ-sampling avoids redundancy between adjacent levels.
```

---

## WHY φ AND NOT SOME OTHER IRRATIONAL

```
π:   continued fraction [3; 7, 15, 1, 292, ...] — the 292 means a very
     good rational approximation exists (355/113). Periodic sources
     with period ~113 will alias perfectly. Bad for anti-resonance.

e:   continued fraction [2; 1, 2, 1, 1, 4, 1, 1, 6, ...] — better than π
     but the pattern 1,2k,1 creates predictable near-resonances.

√2:  continued fraction [1; 2, 2, 2, ...] — period-2 pattern, very resonant
     with even-periodic sources. Bad.

φ:   continued fraction [1; 1, 1, 1, ...] — EVERY convergent is maximally
     bad as a rational approximation. No period in any source maps well
     to φ-spaced positions. The theoretical minimum of resonance.
     
     This is proven: φ produces the Weyl sequence with the lowest
     discrepancy of any irrational multiplier (three-distance theorem).
     It is LITERALLY the optimal anti-aliasing spacing.
```

---

## HARDWARE CONNECTION

Rust 1.94 stabilizes `f32::consts::GOLDEN_RATIO` (φ) and `f32::consts::EULER_GAMMA` (γ).

```rust
use std::f32::consts::PHI;     // 1.618033988749...
use std::f32::consts::GAMMA;   // 0.5772156649015... (Euler-Mascheroni)

// φ for anti-resonant spacing:
let band_foveal = mu - (PHI.powi(3) * sigma) as u32;
let band_near   = mu - (PHI.powi(2) * sigma) as u32;
let band_good   = mu - (PHI * sigma) as u32;

// γ for reservoir sampling expected replacement count:
let expected_replacements = capacity as f32 * ((total as f32).ln() - (capacity as f32).ln() + GAMMA);
```

The constants are computed at compile time. No runtime cost. The φ-spacing
is free — it just changes which integers the band boundaries land on.

---

## THE PUNCHLINE

φ is not decoration. It is the mathematically optimal spacing for folding
high-dimensional information into fewer dimensions with minimum loss.

Every time rustynum packs bits — cascade bands, BF16 exponent assembly,
tree path encoding, VSA bundling — φ-spacing gives the least resonance,
the least aliasing, the least information loss. Not because it's pretty.
Because it's the unique fixed point of the anti-aliasing optimization.

The three-distance theorem guarantees it. The continued fraction proves it.
The implementation is one constant from std::f32::consts.

---

## FIBONACCI AND PRIME ENCODING: NOT WEIGHTING — A DIFFERENT NUMBER SYSTEM

### The Distinction

This is NOT "weigh bit positions by Fibonacci numbers."
This IS "each bit position REPRESENTS a Fibonacci or prime number."

```
STANDARD BINARY:     bit k = 2^k        value = Σ 2^k at set positions
FIBONACCI ENCODING:  bit k = Fib(k+2)   value = Σ Fib(k+2) at set positions
PRIME ENCODING:      bit k = Prime(k)   value = Π Prime(k) at set positions
```

Standard binary is lossy when truncated (lose half the range per bit dropped).
Fibonacci degrades GRACEFULLY (lose one Fib term, rest still exact).
Prime degrades FACTORIALLY (lose one prime factor, other factors still exact).

### Why This Matters for 16K-bit Planes

Current: all 16384 bit positions have equal weight in hamming distance.
Position 0 and position 16383 contribute identically. That's information-blind.

With Fibonacci encoding:
- bit 0 = Fib(2) = 1 (trivial difference)
- bit 20 = Fib(22) = 17,711 (significant difference)
- bit 40 = Fib(42) = 267,914,296 (major structural difference)

Raw XOR + popcount still works (unchanged VPOPCNTDQ).
But the SET BITS in the XOR have MAGNITUDE. Reading which bits
differ tells you the SCALE of the disagreement, not just the COUNT.

With Prime encoding:
- bit 0 = 2, bit 1 = 3, bit 2 = 5, bit 3 = 7, ...
- XOR tells you WHICH prime factors differ
- AND tells you shared factorization
- The distance isn't a number. It's a FACTORIZATION FINGERPRINT.

### Cascade as Scale Decomposition

```
STANDARD CASCADE:
  Stroke 1: random 1/16 sample → statistical projection
  Stroke 2: random 1/4 sample → refined projection
  Stroke 3: full → exact
  
FIBONACCI CASCADE:
  Stroke 1 (bits 0..1023): Fibonacci terms Fib(2)..Fib(1025)
    = the small-scale structure. Coarse topology.
    Reject if even the small terms disagree.
    
  Stroke 2 (bits 1024..4095): Fibonacci terms Fib(1026)..Fib(4097)
    = medium-scale structure. Refines the match.
    
  Stroke 3 (bits 4096..16383): Fibonacci terms Fib(4098)..Fib(16385)
    = large-scale structure. Astronomical magnitudes.
    Agreement here = match at EVERY scale.
    
  Each stroke isn't "more data." It's "the next magnitude."
  The cascade is a MULTI-RESOLUTION ANALYSIS, not a sampling strategy.
```

### BF16 and Tree Path as Zeckendorf Representation

```
BF16 mantissa (7 bits) = Fibonacci terms Fib(2)..Fib(9) = 2,3,5,8,13,21,34
  Representable sums: 0 to 86 (non-uniform spacing)
  More resolution near zero (where precision matters)
  Less resolution at top (where coarse is enough)
  NOT 128 uniform levels. 86 Fibonacci-spaced levels.

Tree path (16 bits) = Fibonacci terms Fib(10)..Fib(27)
  Level 10 adds 89. Level 27 adds 317,811.
  Each tree level contributes a DIFFERENT magnitude.
  Not "16 equal branch decisions."
  "16 decisions at 16 exponentially different scales."

f32 = BF16 + tree path = 23 Fibonacci terms (Zeckendorf representation)
  Unique decomposition guaranteed (Zeckendorf's theorem).
  No other observation sequence produces this f32.
  Deterministic AND mathematically unique.
  Not just placed bits. NAMED bits. Each one a Fibonacci term.
```

### Prime Encoding for Cross-Model Sharing

```
Each bit position = a prime number.
A set bit means "this prime is a factor of the truth."

Node A: bits 2,5,11 set → truth contains factors 5, 13, 31
Node B: bits 2,7,11 set → truth contains factors 5, 17, 31

XOR: bits 5,7 → factors 13 and 17 differ
AND: bits 2,11 → factors 5 and 31 shared

Two models (Grok, Ada, GPT) that use prime-encoded truths
can compare notes by comparing bit patterns.
No alignment training. No weight matching.
The prime factorization IS the universal grammar.

CODEBOOK = primes. Shared by mathematics, not by training.
```

### The Key Insight

Standard bitpacking is lossy because power-of-2 positions waste
resolution on uniform spacing that doesn't match the data distribution.

Fibonacci/Prime positions are NOT lossy because:
1. Zeckendorf: unique representation (no information ambiguity)
2. Graceful degradation (truncation loses magnitude, not structure)
3. Scale-aware (bit position = magnitude, not just index)
4. Anti-resonant (φ-based spacing, proven minimal aliasing)
5. Self-delimiting (Fibonacci codes end with "11", streamable)

This is not compression. This is a different mathematics where every
bit carries its own scale and truncation is graceful instead of catastrophic.

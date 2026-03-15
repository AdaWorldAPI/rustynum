# CASCADE_TETRIS.md

## Tetris Cascade: Incremental Strokes + Prefetch Interleaving + Typed array_chunks

**Status:** Architecture spec. Builds on BELICHTUNGSMESSER.md.
**Prereqs:** Session C (hdr::Cascade), Rust 1.94 (array_chunks stable).

---

## THE PROBLEM WITH CURRENT CASCADE

Current cascade samples OVERLAPPING regions of the vector:

```
Stroke 1: every 16th word across full 2048 bytes → reads 128 bytes
Stroke 2: every 4th word across full 2048 bytes  → reads 512 bytes
Stroke 3: full 2048 bytes

Stroke 2 RE-READS the bytes from Stroke 1. Wasted bandwidth.
Stroke 3 RE-READS everything. Double wasted.
Total bytes read: 128 + 512 + 2048 = 2688 (640 bytes redundant)
```

---

## TETRIS STROKES: ZERO REDUNDANCY

Each stroke is a SLICE of the vector that fills the gap left by previous strokes.
Like Tetris pieces filling open rows on the memory bus:

```
Stroke 1: bytes [0..128]       128 bytes fresh   → project × 16
Stroke 2: bytes [128..512]     384 bytes fresh   → cumulative with Stroke 1
Stroke 3: bytes [512..2048]   1536 bytes fresh   → cumulative with 1+2

Total bytes read: 128 + 384 + 1536 = 2048. EXACT. No byte read twice.
```

The projection is CUMULATIVE:

```rust
let d1 = hamming(&q[..128], &c[..128]);
let projected_1 = d1 * 16;                    // full-vector estimate from 1/16
if projected_1 > reject_threshold { continue; }  // ~84% rejected

let d2 = hamming(&q[128..512], &c[128..512]);
let cumulative_2 = d1 + d2;
let projected_2 = cumulative_2 * 4;           // full-vector estimate from 1/4
if projected_2 > reject_threshold { continue; }  // ~90% of survivors rejected

let d3 = hamming(&q[512..], &c[512..]);
let exact = d1 + d2 + d3;                     // EXACT. d1+d2+d3 = full hamming.
```

---

## ARRAY_CHUNKS (RUST 1.94): TYPED STROKES

`array_chunks::<N>()` gives `&[u8; N]` — compile-time sized. The compiler
monomorphizes each stroke separately:

```rust
fn cascade_stroke_1(q: &[u8; 2048], c: &[u8; 2048]) -> u32 {
    let q1: &[u8; 128] = q[..128].try_into().unwrap();
    let c1: &[u8; 128] = c[..128].try_into().unwrap();
    simd::hamming_distance(q1, c1) as u32
    // Compiler knows 128 bytes → exactly 2 VPOPCNTDQ iterations
    // No loop counter. No branch. Just 2 instructions.
}

fn cascade_stroke_2(q: &[u8; 2048], c: &[u8; 2048], d1: u32) -> u32 {
    let q2: &[u8; 384] = q[128..512].try_into().unwrap();
    let c2: &[u8; 384] = c[128..512].try_into().unwrap();
    d1 + simd::hamming_distance(q2, c2) as u32
    // Compiler knows 384 bytes → exactly 6 VPOPCNTDQ iterations
}

fn cascade_stroke_3(q: &[u8; 2048], c: &[u8; 2048], d12: u32) -> u32 {
    let q3: &[u8; 1536] = q[512..].try_into().unwrap();
    let c3: &[u8; 1536] = c[512..].try_into().unwrap();
    d12 + simd::hamming_distance(q3, c3) as u32
    // Compiler knows 1536 bytes → exactly 24 VPOPCNTDQ iterations
}
```

Each stroke: known size → known unroll count → zero runtime overhead.

---

## PREFETCH INTERLEAVING

While VPOPCNTDQ crunches Stroke 1 of candidate A, prefetch Stroke 2 data
for the PREVIOUS survivor. The memory load and compute overlap:

```
SEQUENTIAL (wasted bandwidth):
CPU:  [POPCNT A:s1][wait][POPCNT A:s2][wait][POPCNT A:s3]
MEM:  [load A:0..128]  [idle]  [load A:128..512]  [idle]  [load A:512..2048]

INTERLEAVED (bandwidth filled):
CPU:  [POPCNT A:s1][POPCNT B:s1][POPCNT C:s1][POPCNT prev_survivor:s2]
MEM:  [prefetch B:0..128][prefetch C:0..128][prefetch survivor:128..512]
```

```rust
fn cascade_batch_tetris(
    query: &[u8; 2048],
    candidates: &[[u8; 2048]],  // array_chunks::<2048>() from the database
    bands: &[u32; 4],
) -> Vec<(usize, u32)> {
    let mut survivors_s1: Vec<(usize, u32)> = Vec::new();
    let mut results: Vec<(usize, u32)> = Vec::new();

    // Pass 1: Stroke 1 on ALL candidates + prefetch for survivors
    for (i, cand) in candidates.iter().enumerate() {
        // Prefetch next candidate's Stroke 1
        if i + 1 < candidates.len() {
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    candidates[i + 1].as_ptr() as *const i8,
                    core::arch::x86_64::_MM_HINT_T0
                );
            }
        }

        let d1 = simd::hamming_distance(&query[..128], &cand[..128]) as u32;
        if d1 * 16 <= bands[2] {
            survivors_s1.push((i, d1));
            // Prefetch this survivor's Stroke 2 region
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    cand[128..].as_ptr() as *const i8,
                    core::arch::x86_64::_MM_HINT_T0
                );
            }
        }
    }

    // Pass 2: Stroke 2+3 on survivors (data already prefetched)
    for &(i, d1) in &survivors_s1 {
        let cand = &candidates[i];
        let d2 = simd::hamming_distance(&query[128..512], &cand[128..512]) as u32;
        let cumulative = d1 + d2;
        if cumulative * 4 > bands[1] { continue; }

        let d3 = simd::hamming_distance(&query[512..], &cand[512..]) as u32;
        let exact = d1 + d2 + d3;
        results.push((i, exact));
    }

    results
}
```

---

## L1 CACHE BOUNDARY

All strokes must stay within L1 (32KB):

```
Stroke 1: 128 bytes    ← deep in L1
Stroke 2: 384 bytes    ← deep in L1
Stroke 3: 1536 bytes   ← fits in L1
Full vector: 2048 bytes ← fits in L1

NEVER concatenate vectors for bulk processing.
64KB = L1 size → 12x SLOWER than scalar (measured).
Per-plane operation is the ONLY design that scales.
```

See L1_CACHE_BOUNDARY.md for the full analysis.

---

## PERFORMANCE ESTIMATE

```
Current cascade (overlapping samples):
  128 + 512 + 2048 = 2688 bytes read per candidate
  ~84% rejected at stroke 1 → 16% reach stroke 2
  ~90% of survivors rejected at stroke 2 → ~1.6% reach stroke 3
  
Tetris cascade (incremental slices):
  128 + 384 + 1536 = 2048 bytes read per candidate (exact, no waste)
  Same rejection rates (same statistical power, different byte positions)
  Save 640 bytes of redundant reads per candidate that survives to stroke 2
  
At 1M candidates:
  Stroke 1: 1M × 128 bytes = 128MB → ~1.2ms
  Stroke 2: 160K × 384 bytes = 61MB → ~0.6ms (survivors only)
  Stroke 3: 16K × 1536 bytes = 24MB → ~0.2ms (survivors only)
  
  Total: ~2ms (vs ~5ms with overlapping samples)
  
With prefetch interleaving: ~30% further reduction from hiding latency
  Estimated: ~1.4ms for 1M candidates
```

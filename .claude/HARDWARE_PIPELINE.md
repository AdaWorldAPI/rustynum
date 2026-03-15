# HARDWARE_PIPELINE.md

## Every Step Has a Hardware Instruction

**Status:** Reference. Maps the full pipeline to silicon.
**Rust version:** 1.94 stable (all intrinsics listed are stable).

---

## THE PIPELINE

```
STEP                        INSTRUCTION              STABLE SINCE    LATENCY
────────────────────────────────────────────────────────────────────────────────
XOR two 16K planes          VPXORD (__m512i)          Rust 1.89       1 cycle
Popcount 16K XOR result     VPOPCNTDQ (__m512i)       Rust 1.89       3 cycles
Band classify distance      CMP (integer)             always          1 cycle
Accumulate encounter        ADD (integer)             always          1 cycle
Alpha threshold check       CMP (integer)             always          1 cycle

f32 → BF16 (leaf insert)   VCVTNEPS2BF16             Rust 1.89       4 cycles
                            (avx512bf16)              (hardware rounding)

BF16 → f32 (hydrate)       hardware conversion        Rust 1.94       3 cycles
                            (avx512fp16 intrinsics)

BF16 dot product            VDPBF16PS                 Rust 1.89       5 cycles
(truth comparison)          (avx512bf16)              (32 BF16 pairs → f32)

Prefetch next candidate     _MM_PREFETCH T0           always          0 (async)
Branch prediction (tier)    predicted branch           always          0 (after 1st)

CPU feature detection       is_x86_feature_detected!  always          ~1 cycle
(LazyLock, first call)      (cached atomic load)                      (after init)
```

---

## SAFE INTRINSICS IN RUST 1.94

All intrinsics that do NOT take pointer arguments are safe when called
from a function with `#[target_feature(enable = "...")]`:

```rust
#[target_feature(enable = "avx512f")]
fn example() {
    // SAFE (no pointers):
    let a = _mm512_set1_ps(1.0);         // creates value
    let b = _mm512_add_ps(a, a);          // register → register
    let c = _mm512_reduce_add_ps(a);      // register → scalar
    let d = _mm512_xor_si512(x, y);       // register → register
    let e = _mm512_popcnt_epi64(x);       // register → register

    // STILL UNSAFE (raw pointers):
    let f = unsafe { _mm512_loadu_ps(ptr) };      // ptr → register
    let g = unsafe { _mm512_storeu_ps(ptr, a) };   // register → ptr
}
```

Impact: LLVM can inline through safe calls within `#[target_feature]` functions.
The `unsafe` boundary in PR #102 is what caused the 24% sdot regression —
the compiler couldn't inline through it.

---

## RUST 1.94 STABILIZED FEATURES FOR THIS ARCHITECTURE

```
FEATURE                  USE IN PIPELINE                         IMPACT
──────────────────────────────────────────────────────────────────────────
avx512fp16 intrinsics    BF16→f32 hydration                     Hardware truth hydration
array_windows<N>()       Typed cascade strokes                   Monomorphized, zero overhead
array_chunks<N>()        Database as typed 2KB chunks            Same
LazyLock                 Tier detection (replaces OnceLock)      Cleaner init, force_mut for tests
LazyLock::force_mut      Override tier in tests                  Test all paths on any hardware
f32::consts::PHI         Golden ratio for φ-spaced bands         No manual constant
f32::consts::GAMMA       Euler-Mascheroni for reservoir stats    No manual constant
Safe intrinsics          Remove unsafe blocks in SIMD code       LLVM can inline through
__cpuid_count safe       CPU feature detection cleaner           Minor cleanup
```

---

## THE FULL RL STEP IN HARDWARE

```rust
/// One RL step: observe, classify, pack, insert, hydrate, learn.
/// Every operation mapped to a hardware instruction.
fn rl_step(
    node_a: &Node,       // 3 × 2KB planes
    node_b: &Node,
    tree: &mut ClamTree,
) -> f32 {
    // ─── OBSERVE: 7 hamming distances ─────────────────────
    // VPXORD + VPOPCNTDQ per plane pair (3 cycles each)
    let d_s   = simd::hamming_distance(node_a.s.bits(), node_b.s.bits());
    let d_p   = simd::hamming_distance(node_a.p.bits(), node_b.p.bits());
    let d_o   = simd::hamming_distance(node_a.o.bits(), node_b.o.bits());
    let d_sp  = d_s + d_p;  // ADD, 1 cycle
    let d_so  = d_s + d_o;
    let d_po  = d_p + d_o;
    let d_spo = d_s + d_p + d_o;

    // ─── CLASSIFY: 7 band assessments ─────────────────────
    // CMP (integer), 1 cycle each
    let exp_bit_1 = (d_s   < foveal_or_near_threshold) as u8;
    let exp_bit_2 = (d_p   < foveal_or_near_threshold) as u8;
    let exp_bit_3 = (d_o   < foveal_or_near_threshold) as u8;
    let exp_bit_4 = (d_sp  < foveal_or_near_threshold) as u8;
    let exp_bit_5 = (d_so  < foveal_or_near_threshold) as u8;
    let exp_bit_6 = (d_po  < foveal_or_near_threshold) as u8;
    let exp_bit_7 = (d_spo < foveal_or_near_threshold) as u8;

    // ─── PACK: BF16 assembly ──────────────────────────────
    // Bit manipulation, ~5 cycles
    let sign: u16 = 0; // causing (query → candidate direction)
    let exponent: u16 = (exp_bit_7 as u16) << 7
                      | (exp_bit_6 as u16) << 6
                      | (exp_bit_5 as u16) << 5
                      | (exp_bit_4 as u16) << 4
                      | (exp_bit_3 as u16) << 3
                      | (exp_bit_2 as u16) << 2
                      | (exp_bit_1 as u16) << 1;
    let mantissa: u16 = finest_distance_normalized(d_p, bands); // 7 bits
    let bf16_bits: u16 = (sign << 15) | (exponent << 7) | mantissa;

    // ─── INSERT: tree leaf ────────────────────────────────
    // f32 → BF16: VCVTNEPS2BF16, 4 cycles
    let leaf_id = tree.insert_bf16(bf16_bits);
    let path_bits = tree.path_to(leaf_id); // 16 bits, integer comparisons

    // ─── HYDRATE: BF16 + path → f32 ──────────────────────
    // BF16→f32 hardware conversion, 3 cycles
    // Then OR with path bits, 1 cycle
    let base_f32_bits = (bf16_bits as u32) << 16; // hardware-assisted
    let full_f32 = f32::from_bits(base_f32_bits | path_bits as u32);

    // ─── LEARN: credit assignment from exponent ───────────
    // Read bits + encounter(), integer accumulation
    if exp_bit_2 == 1 { node_a.p.encounter_toward(&node_b.p); }  // reward P
    if exp_bit_1 == 0 { node_a.s.encounter_away(&node_b.s); }    // punish S
    // ... 8 conditional encounters, ~200ns each

    full_f32  // deterministic ground truth
}
```

Total hardware cost per pair: ~2.8μs.
1M candidates with 0.3% survival: 3000 × 2.8μs = 8.4ms per RL epoch.

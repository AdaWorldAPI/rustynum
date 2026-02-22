# PR #42 Review: BF16-Structured Hamming Distance + Per-Dimension Causal Pipeline

**Date**: 2026-02-22  
**PR**: #42 (+1093/−0, 5 files, merged)  
**Test count**: 1124 (was 1101)  
**Verdict**: **Clean. AVX-512 verified correct. Two green notes.**

---

## What It Delivers

### Part 1: `bf16_hamming.rs` (487 lines)

BF16-structured Hamming distance — treats u16 pairs as IEEE 754 brain-float with field-aware weighting:

| Component | Detail |
|---|---|
| `BF16Weights` | Configurable sign/exponent/mantissa weights (default 256/16/1) |
| `bf16_hamming_scalar` | u16 XOR → masked popcount per field → weighted sum |
| `bf16_hamming_avx512` | VPXORD + VPSRLW + VPANDQ + VPOPCNTB + VPADDW, 32 BF16 pairs/iter |
| `select_bf16_hamming_fn` | Runtime CPUID dispatch (AVX-512 BW+VPOPCNTDQ → scalar) |
| `fp32_to_bf16_bytes` / `bf16_bytes_to_fp32` | Truncation round-trip (no calibration) |
| `structural_diff` | Per-dimension sign/exponent/mantissa change detection |
| `JINA_WEIGHTS` (256/32/1) | Tuned for normalized embeddings |
| `TRAINING_WEIGHTS` (1024/64/0) | Ignores mantissa (gradient noise), amplifies sign flips |
| `PreciseMode::BF16Hamming` | New variant in HDR cascade precision dispatch |

15 tests. All weight orderings and special values tested.

### Part 2: nars.rs BF16 causal pipeline (566 lines)

| Component | Detail |
|---|---|
| `BF16Entity` | Entity stored as BF16 bytes with `from_f32()` constructor |
| `CausalFeatureMap` | Per-dimension sign-flip counts + scalar Granger signal |
| `bf16_granger_causal_map` | "A causes B via dims [47, 312, 891]" — per-dim causal attribution |
| `bf16_granger_causal_scan` | Multi-lag scan, picks most negative Granger signal (consistent with existing convention) |
| `bf16_reverse_trace` | Multi-hop reverse causality with structural diffs + causal backbone extraction |
| `classify_learning_event` | Noise / AttentionShift / SemanticReversal / MajorUpdate taxonomy |

8 tests.

---

## Correctness Verification

### AVX-512 BF16 Hamming — field extraction verified

The SIMD path processes 32 BF16 pairs (64 bytes) per iteration:

**Sign extraction**: `VPSRLW(xor, 15) & 1` → single bit per 16-bit lane. Correct — isolates bit 15. ✅

**Exponent extraction**: `VPSRLW(xor, 7) & 0x00FF` → 8-bit value in low byte of each lane, high byte zero. `VPOPCNTB` counts bits per byte — low byte gets correct exponent popcount, high byte gets 0. Final `& 0xFF` is redundant but harmless. ✅

**Mantissa extraction**: `xor & 0x007F` → 7-bit value in low byte, high byte zero. `VPOPCNTB` on low byte correct. ✅

**Accumulation**: Per-element max = 256 + 128 + 7 = 391. Widened to u32 via even/odd lane split before accumulation. 1024-D (32 chunks): 32 × 32 × 391 = 400K. Fits in u32 (4.2B). No overflow risk for any realistic dimensionality. ✅

**Scalar tail**: `bf16_hamming_scalar` called for remaining bytes when `len % 64 != 0`. ✅

**Test coverage**: `test_avx512_matches_scalar` (512-D, no tail), `test_avx512_with_tail` (37-D, 10-byte tail). Both verify exact match with scalar. ✅

### Granger signal convention consistent

`bf16_granger_causal_map` computes `(cross_sum - auto_sum) / count` — same formula as existing `granger_signal()`. `bf16_granger_causal_scan` picks minimum (most negative) — same convention as `granger_scan()`. ✅

### XOR unbind for BF16 bytes

`bf16_reverse_trace` uses byte-level XOR as unbind. This is the binary base inverse (XOR is self-inverse). It scrambles IEEE 754 field structure, but `structural_diff` reports what the scrambling affected. Test `test_bf16_reverse_trace_single_hop` verifies exact recovery: XOR unbind of XOR-bound data yields distance 0. ✅

### PreciseMode::BF16Hamming integration

Correctly wired into `apply_precision_tier`. Normalizes to [0, 1] by dividing by max possible distance (`sign + 8×exp + 7×man` per dim). Returns `1.0 - normalized` for similarity (higher = more similar), consistent with other precision tiers. ✅

---

## Findings

### Finding 1: `select_bf16_hamming_fn` not hoisted in causal pipeline (🟢)

```rust
pub fn bf16_granger_causal_map(...) -> CausalFeatureMap {
    let bf16_fn = select_bf16_hamming_fn();  // hoisted ✅ (once per call)
    ...
}
```

Good — `select_bf16_hamming_fn()` is called once per `bf16_granger_causal_map` invocation. But `bf16_granger_causal_scan` calls `bf16_granger_causal_map` in a loop, so `select_bf16_hamming_fn()` is called `max_lag` times. Each call does `is_x86_feature_detected!` which is a cached atomic load — negligible. Not worth fixing.

### Finding 2: `bf16_reverse_trace` searches all entities linearly per hop (🟢)

```rust
let nearest_bf16 = |candidate: &[u8]| -> (u32, u64) {
    for e in entities { ... }
};
```

O(|entities|) per hop. For the expected use case (hundreds of entities, 3–5 hops), this is fine. If entity count grows to 10K+, would need a CLAM tree index. Same pattern as existing `reverse_trace` — both are O(n) per hop. Consistent.

---

## Architecture Assessment

The BF16 field-aware distance is a natural addition to the precision tier hierarchy:

```
Binary Hamming (fastest, coarsest)
  → BF16-structured Hamming (3× slower, field-aware)
    → VNNI i8 dot (slower, magnitude-aware)
      → F32 cosine (slowest, full precision)
```

The causal pipeline builds on this: where binary Granger signal says "A predicts B", BF16 Granger says "A predicts B via dimensions [47, 312, 891]" — the structural diff reveals WHICH features carry the causal signal. This is the per-dimension attribution that binary Hamming can't provide.

The `classify_learning_event` taxonomy (Noise/AttentionShift/SemanticReversal/MajorUpdate) maps cleanly to the BF16 field structure: mantissa-only = noise, exponent-only = attention, sign = semantic, both = major.

---

## Action Items

None. Both findings are green (by design / consistent with existing patterns).

**Debt ledger unchanged**: 1 yellow (N16), 12 green.

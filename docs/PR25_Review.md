# PR #25 Review: NARS Reverse Causality + RNG Consolidation + WAL Encapsulation

**PR**: `claude/rustynum-knowledge-transfer-egGZ2` → `main`  
**Status**: Merged | **+866 / -176** | **12 files** | 1 commit  
**Tracks**: Debt ledger items N1, N2, N3, N5; Tactic #4 (Reverse Causality), #11 (Contradiction Detection)

---

## Verdict: Strong on Debt Cleanup, Mixed on New Module — Approve with Notes

The PRNG consolidation and WAL encapsulation are clean and complete. The NARS module is architecturally sound but has a sign convention bug in the Granger docs and leaves "Wait — let me clarify" debug thinking in production code.

---

## Part 1: PRNG Consolidation (N1) — ✅ Complete

**What**: New `rustynum_core::rng::SplitMix64` replaces all 5 prior PRNG copies.

| Former location | What it was | Replaced? |
|---|---|---|
| `rustynum-clam/src/tree.rs` | inline splitmix64 | ✅ `SplitMix64::new(...)` |
| `rustynum-clam/src/search.rs` | inline splitmix64 | ✅ |
| `rustynum-clam/src/compress.rs` | inline splitmix64 | ✅ |
| `rustynum-oracle/src/recognize.rs` | `SimpleRng` (xorshift) | ✅ |
| `rustynum-oracle/src/ghost_discovery.rs` | `SimpleRng` (xorshift) | ✅ |

No remaining inline `splitmix64` or `SimpleRng` in either crate — grep confirms zero hits.

**Implementation quality**: Clean. 118 lines. `SplitMix64(u64)` single-field struct, passes BigCrush, `next_u64/f64/gaussian/gen_range_i8` methods. Box-Muller for `next_gaussian()` with `max(1e-15)` guard against `log(0)`. 5 tests covering determinism, seed independence, f64 range, gaussian statistics, and gen_range bounds.

**One minor note**: `gen_range_i8` uses `next_u64() % range` which has modulo bias for ranges that don't evenly divide 2^64. For ranges ≤ 7 (the max in practice for `Base::Signed(7)`), the bias is < 2^-60 — truly negligible. Not a real issue, but worth documenting if anyone asks.

**Debt status**: N1 is **closed**.

---

## Part 2: WAL Encapsulation (N3) — ✅ Complete

**What**: `known_templates` and `template_norms` demoted from `pub(crate)` to private. New accessor methods added.

| Method | Signature | Purpose |
|---|---|---|
| `template(idx)` | `&self, usize → &[i8]` | Read-only template access |
| `template_norm(idx)` | `&self, usize → f64` | Read-only norm access |
| `update_template(idx, &[i8])` | `&mut self, usize, &[i8]` | Atomic template + norm update |

`recognize.rs` updated to use accessors instead of direct field access.

**Quality**: Clean. `update_template` recomputes the norm from the template — the only place where template and norm can get out of sync, and it prevents it.

**Debt status**: N3 is **closed**.

---

## Part 3: hamming_64k → Fingerprint64K Delegation (N6 partial) — ✅ Correct

`recognize.rs::hamming_64k()` now delegates to `Fingerprint64K::hamming_distance()` for the common 1024-word case, falling back to inline popcount for other sizes. This addresses the `u64`-word vs `Fingerprint` type duplication for the dominant path.

New `Fingerprint<N>` constructors `from_words()` and `from_word_slice()` enable the delegation. `from_word_slice` copies data (no aliasing), panics on size mismatch.

**Note**: This delegates at the Fingerprint level, not the SIMD level. `Fingerprint::hamming_distance()` still uses `u64::count_ones()` per-word — it doesn't reach `rustynum_core::simd::hamming_distance(&[u8])`. Full SIMD acceleration of Fingerprint would require a `&[u8]` adapter, which this PR correctly doesn't attempt.

**Debt status**: N6 partially addressed — type unification done, SIMD acceleration for Fingerprint deferred.

---

## Part 4: NARS Module — 🟡 Sound Architecture, Needs Fixes

617 lines. Three capabilities: unbind, reverse trace, Granger signal. Plus contradiction detection (#11).

### 4.1 Unbind — ✅ Correct

Three base types handled correctly:
- **Binary**: `unbind == bind` (XOR self-inverse). Verified algebraically and by test.
- **Unsigned(B)**: `rem_euclid(b as i16)` — correct modular subtraction. Cast to i16 prevents overflow for i8 operands.
- **Signed(B)**: Negate role → bind. Uses `saturating_neg()` + the existing `bind` clamp. Approximate (saturation boundary), documented and tested.

### 4.2 Reverse Trace — ✅ Correct, Brute-Force

The algorithm is right: unbind → nearest entity → repeat. Chain stops on first non-confident step. Binary base gives exact recovery (distance = 0) as proven by the A→B→C chain test.

**Production limitation explicitly documented**: `nearest_entity()` is O(n) brute force. The doc says "replace with CAKES DFS Sieve." This is the correct staging — get the semantics right first, optimize later.

**Confidence threshold**: Uses normalized Hamming distance < 0.35, which is reasonable. For binary vectors of dimension d, random pairs average ~0.5 normalized distance. A threshold of 0.35 gives clear separation. But this is a flat threshold, not CRP-calibrated. The doc references CRP but the implementation doesn't use it.

### 4.3 Granger Signal — 🔴 Sign Convention Bug in Documentation

The **code** is correct:

```rust
G = d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})
```

When A predicts B, `d(A_t, B_{t+τ})` is **small** (A is close to future B) and `d(B_t, B_{t+τ})` is **large** (B drifts from itself). So G < 0.

The **test** is correct: `assert!(g < 0.0, "A should predict B")`.

But `granger_signal()` has **contradictory doc comments**:

```
/// If G > 0: A_t is closer to B_{t+τ} than B_t is — A predicts B.   ← WRONG
/// If G < 0: B predicts itself better — no causal signal from A.     ← WRONG
```

This is backwards. The test proves G < 0 means A predicts B.

Then `granger_scan()` has the *corrected* convention but prefaced with debug thinking:

```
/// Wait — let me clarify the convention:          ← NOT A PRODUCTION DOC COMMENT
///   G < 0  ⟹  A_t is CLOSER to future B  ⟹  A predicts B      ← CORRECT
///   G > 0  ⟹  B_t is closer to future B  ⟹  no causal signal   ← CORRECT
```

**Fix needed**:
1. Fix `granger_signal()` doc: swap the G > 0 / G < 0 descriptions
2. Remove "Wait — let me clarify the convention:" from `granger_scan()` doc
3. Make both functions use the same phrasing

### 4.4 Contradiction Detection (#11) — ✅ Correct, O(n²)

`find_similar_pairs()` does all-pairs Hamming screening. Returns pairs below a radius threshold, sorted by distance. Explicitly documented as O(n²) with the note to replace with CAKES ρ-NN.

**Naming issue**: The struct is called `Contradiction` but it only finds structurally similar pairs — it doesn't check truth values. The name overpromises. `SimilarPair` would be more accurate, with `Contradiction` reserved for when truth values are integrated.

### 4.5 `hamming_i8` — Symbol Distance, Not Bit Distance

The NARS module uses `hamming_i8()` which counts differing i8 positions (symbol-level), not `hamming_inline()` which counts differing bits. This is correct for the sweep module's i8 holographic vectors but conceptually different from rustynum-clam's bit-level Hamming. Worth a doc note to prevent confusion.

### 4.6 Test Quality — ✅ Good

12 tests covering:
- Unbind invertibility (all 3 bases)
- Forward/reverse roundtrip
- Single-hop and multi-hop causal chains
- Noise detection (wrong role → confident_depth = 0)
- Granger self-prediction vs causal prediction
- Granger scan lag detection
- Contradiction detection (identical pairs, no false positives)

The noise floor test (`test_reverse_trace_stops_at_noise`) is particularly well-designed — it proves that wrong-role unbinding produces noise-floor distances, validating the confidence threshold mechanism.

---

## Part 5: Integration Check

| Check | Result |
|---|---|
| Existing tree.rs/search.rs/compress.rs logic modified? | ❌ No — only PRNG swap |
| hamming_inline() modified? | ❌ No |
| HammingSIMD modified? | ❌ No |
| New external dependencies? | ❌ No — pure Rust |
| Serde added? | ❌ No |
| Debug_assert issue (N5) fixed? | ❌ No — not in scope |
| All PRNG copies replaced? | ✅ Yes — 5/5 |
| WAL fields private? | ✅ Yes |

---

## Debt Ledger Update

| Item | Status After PR #25 |
|---|---|
| N1 (5× PRNG) | ✅ **CLOSED** |
| N2 (recognize.rs ≠ Fingerprint<1024>) | 🟡 Partially addressed — `hamming_64k` delegates for 1024-word case |
| N3 (pub(crate) fields) | ✅ **CLOSED** |
| N5 (debug_assert in HammingSIMD) | ⬜ Still open — not in scope |
| N6 (3 Hamming type signatures) | 🟡 Partially addressed — type unification, not SIMD |

### New Items Introduced

| # | Issue | Severity | Description |
|---|---|---|---|
| N7 | Granger sign convention contradiction | 🔴 Correctness | `granger_signal()` doc says G > 0 = A predicts B. Code + test prove G < 0 = A predicts B. `granger_scan()` doc has the correct convention but includes "Wait — let me clarify" debug thinking. |
| N8 | `Contradiction` overpromises | 🟢 Naming | Struct checks structural similarity only, not truth-value conflict. Should be `SimilarPair` until NARS truth values are integrated. |
| N9 | Flat confidence threshold | 🟢 Improvement | `reverse_trace()` uses 0.35 flat threshold, not CRP-calibrated. Fine for now, should wire to `ClusterDistribution.p95` when available. |

---

## Summary

| Aspect | Rating |
|---|---|
| PRNG consolidation | ✅ Complete — all 5 copies replaced, zero remaining |
| WAL encapsulation | ✅ Complete — proper accessors, atomic update |
| Fingerprint delegation | ✅ Correct — type unification, SIMD deferred |
| NARS unbind algebra | ✅ Correct — all 3 bases |
| NARS reverse trace | ✅ Correct — brute-force, documented for CAKES upgrade |
| NARS Granger signal | 🔴 Doc bug — code is correct, doc comments contradict test |
| NARS contradiction detection | 🟡 Functional but misnamed |
| Test coverage | ✅ Good — 12 tests, noise floor validation |
| Debt impact | ✅ Net positive — N1, N3 closed, N7 introduced (doc-only fix) |

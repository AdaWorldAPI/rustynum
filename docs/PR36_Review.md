# PR #36 Review: PreciseMode — 5-Path Stroke 3 for HDR Cascade

**Merged review** — two independent reviews cross-referenced  
**Date**: 2026-02-22  
**PR**: #36 (+525/−38, 6 files) on top of #35 (+487, 3-stroke batch engine)  
**Verdict**: **Merge. Best PR pair in the repo's history.** Five findings below, one yellow.

---

## What It Delivers

PRs #35–36 together close the three bugs from the design doc:

| Bug | Fix | Verified |
|---|---|---|
| Scalar path (no SIMD in adaptive search) | `select_hamming_fn()` hoisted dispatch, `hdr_cascade_search` uses fn pointer | ✅ `hamming_chunk_inline` eliminated from search path |
| No type transition in Tier 3 | `PreciseMode` enum: Off/Vnni/F32/BF16/DeltaXor | ✅ 5 distinct Stroke 3 paths |
| Serial per-candidate loop | 3-stroke batch pipeline: warmup → partial reject → incremental full → precision | ✅ Survivors/finalists pattern |

New APIs: `hdr_search`, `hdr_search_f32`, `hdr_search_delta`, `CogRecord::hdr_sweep`.  
Python bindings for all four. 13 new tests, 1070 total.  
Backward compatible — `hamming_search_adaptive` delegates to `hdr_cascade_search` with `PreciseMode::Off`.

---

## Finding 1: σ approach — better than spec (✅)

The implementation improved on the design doc's pure population-σ:

```rust
let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();
let sigma_pop = var.sqrt();
let sigma = sigma_est.max(sigma_pop).max(1.0);
```

`sigma_est` is the binomial sampling error at the threshold boundary — the correct statistical quantity for where false rejections happen. `sigma_pop` from 128-sample warmup captures the empirical spread. Taking the max prevents under-rejection in both regimes: tight populations (σ_est dominates) and threshold-clustered distributions (σ_pop dominates). Floor at 1.0 prevents zero-sigma degenerate case.

Sound. No action needed.

---

## Finding 2: Python bindings crash on bad input (🟡 — file as issue)

Three of four HDR Python bindings pass straight through to Rust `assert_eq!` without validation:

```rust
// hdr_search, hdr_search_f32, hdr_search_delta — NO validation
fn hdr_search(&self, database: PyRef<PyNumArrayU8>, ...) -> PyResult<...> {
    Ok(self.inner.hdr_search(&database.inner, vec_len, count, threshold))
}

// hdr_sweep — HAS validation ✅
fn hdr_sweep(&self, database: Vec<u8>, n: usize, ...) -> PyResult<...> {
    if database.len() != n * 8192 {
        return Err(PyValueError::new_err(...));
    }
    ...
}
```

If `database.len() != vec_len * count`, Rust panics → Python segfault. This is the same N18 pattern from the debt ledger (PyO3 assert crashes). The fix is identical: validate before calling inner, return `PyValueError`.

**Severity**: 🟡 — anyone calling from Python with wrong-sized data crashes the interpreter.

---

## Finding 3: `approx_hamming_candidates` scalar popcount antipattern (🟢 — REFACTOR marker)

```rust
let xor = q ^ d;           // u8x64 SIMD XOR — good
let arr = xor.to_array();  // spill 64 bytes to stack — bad
for byte in arr {
    dist += byte.count_ones();  // 64 scalar popcounts — defeats VPOPCNTDQ
}
```

Called from exactly one test. Not on any hot path. But it's a trap for anyone who adds a caller — the comment says "SIMD XOR + popcount" when only the XOR is SIMD. Should get a `// REFACTOR:` marker or delegate to `hamming_batch`.

---

## Finding 4: s1_bytes not 64-byte aligned for non-power-of-2 vectors (🟢 — cosmetic)

```rust
let s1_bytes = (vec_bytes / 16).max(64).min(vec_bytes);
```

For `vec_bytes=1500`: s1_bytes=93. The SIMD path handles the tail via scalar fallback, so this is correct — just suboptimal. Wastes up to 63 bytes of SIMD potential on unaligned tail.

Fix: `((vec_bytes / 16).max(64) + 63) & !63` rounds up to 64-byte boundary. Trivial.

Primary use case (2048-byte CogRecord containers) gives s1=128, already aligned. Theoretical issue only.

---

## Finding 5: `hdr_sweep` is serial, not batch-stroke (🟢 — acceptable)

CogRecord's 4-channel sweep uses per-candidate serial cascade, not the batch-stroke model from `hdr_cascade_search`. This is architecturally acceptable because:

- Stage 1 (META) rejects ~90%+ — survivors are too sparse for batch benefit  
- CPUID dispatch is correctly hoisted (`hamming_fn` + `dot_fn` selected once)  
- Each channel is a full 2KB vector — SIMD amortized across 32+ VPOPCNTDQ per candidate  
- Compound early-exit across 4 channels is inherently serial (channel N depends on channel N-1 passing)

No action needed.

---

## VNNI Implementation Correctness

The `dot_i8_vnni` intrinsic uses the signed×signed via unsigned trick correctly:

```
a_unsigned = a_signed XOR 0x80    (shifts [-128,127] to [0,255])
dpbusd_result = Σ(a_unsigned × b_signed)
             = Σ((a_signed + 128) × b_signed)
             = Σ(a_signed × b_signed) + 128 × Σ(b_signed)
∴ signed_result = dpbusd_result - 128 × Σ(b_signed)
```

Code: `result = total_biased - 128 * total_b` — matches the algebra. Scalar tail handles non-64-byte-aligned vectors. ✅

---

## Action Items

| # | Severity | Action | Est |
|---|---|---|---|
| 1 | 🟡 | Add input validation to `hdr_search`, `hdr_search_f32`, `hdr_search_delta` Python bindings | 5 min |
| 2 | 🟢 | Add `// REFACTOR: delegate to hamming_batch` marker on `approx_hamming_candidates` | 1 min |
| 3 | 🟢 | Round s1_bytes to 64-byte boundary | 1 min |

Items 2–3 can ride with the next PR that touches those files. Item 1 should be filed.

---

## Updated Open Debt

### Closed by PRs #29–#36:

N13, N14, N15, N17, N18, N19, N20, N22 (PR #29). CPUID hoisting, type transition, batch stroke, PreciseMode (PRs #35–36).

### Open:

| # | Sev | Item | Notes |
|---|---|---|---|
| N2 | 🟡 | recognize.rs `Vec<u64>` not `Fingerprint<1024>` | |
| N6 | 🟡 | 6 Hamming implementations (now including `approx_hamming_candidates`) | |
| N16 | 🟡 | Hardcoded concept indices in ghost_discovery | |
| N21 | 🟡 | `hamming_slice` in arrow bridge still scalar | |
| U1 | 🟡 | Signed unbind saturation (nars.rs:46-50) | Unfiled |
| U2 | 🟡 | `partial_cmp().unwrap()` NaN panics (recognize.rs, organic.rs) | Unfiled |
| **NEW** | 🟡 | HDR Python bindings no input validation (3 functions) | File |
| N9 | 🟢 | Flat confidence threshold in reverse_trace | |
| N11 | 🟢 | Clone-per-hop in reverse_trace | |
| N23 | 🟢 | LFD integer division | |
| N24 | 🟢 | Unnecessary with_gil() | |
| N26 | 🟢 | Arrow bridge expect() | |
| N27–N30 | 🟢 | Dead code, stale docs, pruned_subtrees zero | |
| U3 | 🟢 | 64K projector memory (by design) | |
| U4 | 🟢 | learn_improves tautology | |
| **NEW** | 🟢 | `approx_hamming_candidates` scalar popcount | Test-only |
| **NEW** | 🟢 | s1_bytes alignment | Cosmetic |
| **NEW** | 🟢 | F32 dequantize loop scalar | Fine for ~200 finalists |

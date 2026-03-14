# PR #102 Review: Runtime ISA Dispatch

**Reviewer:** Claude Code (automated deep review)
**Date:** 2026-03-14
**Verdict:** Mixed. Correct architecture, sloppy execution, one critical performance bug.

---

## TL;DR

The runtime dispatch idea is **good and correctly implemented**. The execution is
**messy**: 24 commits of churn (add/delete/restore files repeatedly), philosophical
commit messages in a production repo, inflated README benchmarks that don't match
reality, a **critical performance regression** in popcount at 64KB, and CI that now
skips testing 3 of the most important crates. The PR does what it says but ships
technical debt alongside.

---

## 1. CRITICAL BUG: Popcount SIMD Cliff at 64KB

Benchmarks reveal a **catastrophic performance collapse** at 65536 bytes:

```
Size      | SIMD            | Scalar         | SIMD vs Scalar
----------|-----------------|----------------|----------------
8192      | 74.65 ns        | 1.57 us        | 21x FASTER  (good)
16384     | 143.84 ns       | 3.07 us        | 21x FASTER  (good)
32768     | 284.14 ns       | 6.08 us        | 21x FASTER  (good)
65536     | 149.89 us       | 12.37 us        | 12x SLOWER  (BUG)
```

At 64KB, SIMD throughput drops from **104 GiB/s to 343 MiB/s** -- a **300x throughput
collapse**. The SIMD path goes from 284 ns at 32KB to 149,890 ns at 64KB (a 528x
wall-clock increase for a 2x data increase). This is either a cache aliasing issue,
a codepath bug in the runtime dispatch, or an allocation in the hot path.

**The naive scalar path is 12x faster than SIMD at this size.** This means any user
processing CogRecord-sized data (65536 bits = 8KB is fine, but concatenated records
at 64KB trigger this) will hit a performance cliff that's invisible without
benchmarking.

**Severity:** Critical. This is a shipped regression that makes the SIMD path
actively harmful at certain data sizes.

---

## 2. GEMM Performance: README Claims vs Reality

### What the README says (16 threads):
```
1024x1024: 138.85 GFLOPS, 10.53x speedup over old path
```

### What the benchmark actually produces (4 threads, this machine):
```
1024x1024: 99.19 GFLOPS, 8.33x speedup over old path
```

### What criterion shows (rustynum_rs matmul vs ndarray/nalgebra):
```
Operation                  | rustynum_rs | ndarray    | nalgebra   | rustynum vs best
500x500 matmul             | 6.24 ms     | 4.10 ms    | 4.05 ms    | 1.54x SLOWER
1000x1000 matmul           | 48.12 ms    | 34.67 ms   | 37.72 ms   | 1.39x SLOWER
```

### What the rustyblas sgemm criterion shows:
```
sgemm/1024: 60.0 ms  =>  ~35.8 GFLOPS
```

**The honest picture:** rustynum's GEMM is 1.4-1.5x slower than ndarray (which uses
OpenBLAS under the hood). The README's claimed 138.85 GFLOPS was measured under
ideal conditions (16 threads, specific hardware) that don't represent typical usage.
The 99.19 GFLOPS on 4 threads is respectable but the gap with OpenBLAS remains
significant.

The README explicitly acknowledges the OpenBLAS gap ("0.14x") but then buries it
under pages of HDC benchmarks where rustynum genuinely dominates. This creates a
misleading overall impression.

---

## 3. Where RustyNum Genuinely Wins

HDC/VSA operations are the project's real strength, and the numbers hold up:

```
Popcount 16KB:     SIMD 144 ns vs scalar 3.07 us  = 21x faster
Popcount 32KB:     SIMD 284 ns vs scalar 6.08 us  = 21x faster
```

The adaptive cascade search, Int8 VNNI paths, and bitwise HDC operations are
legitimately fast and have no equivalent in NumPy/ndarray. These claims in the
README check out.

---

## 4. Code Quality Assessment

### What's Good

- **Dispatch pattern is correct.** Three-tier `AVX-512 -> AVX2 -> scalar` with
  `is_x86_feature_detected!()` + `#[target_feature]` + `#[cfg(target_arch)]` is
  the textbook approach. Safety is properly maintained.
- **Scalar fallbacks exist everywhere.** No CPU will crash.
- **simd_compat.rs cleanup.** 1527 lines properly moved to simd_avx512.rs with a
  4-line deprecation shim. Clean rename.
- **SIMD_Auto.rs design doc.** Despite being a `.rs` file containing zero executable
  code (it's all comments), it clearly explains the dispatch architecture.

### What's Bad

**Commit hygiene is terrible.** 24 commits including:
- "Remove command_set.rs" -> "Restore command_set.rs" -> "Delete command_set.rs" ->
  "Restore command_set.rs" (4 commits that cancel each other out)
- "Fix CI" appears 8 times (commits afa1f30 through a7a785c)
- Two philosophical commits with zero code changes:
  - `103f835`: "Sign bit = causality direction. Broken seal = Staunen"
  - `62e82e5`: "BF16 is a truth encoding: sign=polarity, exponent=confidence.scale"

This should have been squashed to ~3-5 focused commits before merge.

**5,000 lines of session documents in `.claude/`.** Files like
`BF16_MANTISSA_CAUSALITY.md` (461 lines of philosophical musing about BF16 as "truth
encoding") and `BELICHTUNGSMESSER.md` (842 lines) are checked into the repo. These
are personal notes, not documentation. They add cognitive load for anyone reading
the repo.

**hdr.rs (556 lines) added without clear motivation in this PR.** This refactors
the cascade search out of simd.rs. The code is fine, but it's scope creep in a PR
titled "runtime ISA dispatch."

---

## 5. CI Coverage Regression

### Before PR #102:
All crates tested (with feature flags)

### After PR #102:
```yaml
cargo test --workspace \
  --exclude rustynum-rs \
  --exclude rustynum-arrow \
  --exclude rustynum
```

Three crates with the most user-facing code are **excluded from CI**. The rationale
(SIGILL on GitHub Actions runners without AVX-512) is valid, but the solution
(just skip them) is lazy. Alternatives:

1. Self-hosted runner with AVX-512
2. Runtime feature detection in tests (some tests already have this)
3. Conditional compilation of test code
4. Docker container with QEMU emulation for AVX-512

Excluding your primary crate from CI means regressions ship silently. This is how
the popcount bug got through.

---

## 6. Technical Debt Ledger

| Item | Severity | Lines |
|------|----------|-------|
| Popcount 64KB regression | Critical | - |
| 3 crates excluded from CI | High | - |
| README GFLOPS claims don't match benchmarks | Medium | - |
| 5,000 lines of `.claude/` session notes in repo | Low | 4,969 |
| SIMD_Auto.rs is a design doc disguised as `.rs` | Low | 59 |
| 24 #[allow(dead_code)] annotations | Low | 24 |
| 6 TODO/FIXME/HACK in source | Low | 6 |
| simd_compat.rs deprecation shim (should remove after migration) | Low | 4 |
| No AVX2 path for FFT or BF16 GEMM (scalar fallback only) | Low | - |

---

## 7. Performance Summary Table

### Operations Where Benchmarks Confirm README Claims:
| Operation | Claimed | Measured | Verdict |
|-----------|---------|----------|---------|
| Popcount SIMD 8-32KB | ~21x vs scalar | 21x | CONFIRMED |
| GEMM old vs new (1024x1024) | 10.53x speedup | 8.33x (4T) | CLOSE (thread count differs) |
| VML sqrt (65536) | Fast | 18.88 us | REASONABLE |

### Operations Where Benchmarks Contradict README Claims:
| Operation | Claimed | Measured | Verdict |
|-----------|---------|----------|---------|
| GEMM 1024x1024 GFLOPS | 138.85 | 99.19 (4T) / 35.8 (criterion) | INFLATED |
| Popcount SIMD 64KB | Fast | 12x SLOWER than scalar | REGRESSION |
| matmul vs ndarray 1000x1000 | Competitive | 1.39x slower | LOSING |

---

## 8. Verdict

**The core idea is sound.** Runtime ISA dispatch is the right move. The
implementation is technically correct -- safety invariants are maintained, the
dispatch pattern is standard, scalar fallbacks exist.

**The execution is sloppy.** The PR ships a critical performance bug, inflates
benchmark numbers in the README, removes CI coverage for 3 major crates, includes
philosophical commits and 5K lines of personal notes, and has 24 commits of
back-and-forth churn that should have been squashed.

**Recommendations:**
1. **P0:** Fix the popcount 64KB regression before any new features
2. **P0:** Restore CI coverage for rustynum-rs, rustynum-arrow, rustynum
3. **P1:** Update README benchmarks to match actual measured performance
4. **P1:** Move `.claude/` session documents to a wiki or delete them
5. **P2:** Squash commit history for future PRs
6. **P2:** Add AVX2 fallback for FFT butterfly operations

---

*Review generated from: full workspace test run (1,543 pass, 0 fail), GEMM
benchmark, HDC benchmark, BLAS criterion benchmark, MKL criterion benchmark,
array_benchmarks criterion, full diff analysis of 89 changed files (+8,650/-3,175
lines).*

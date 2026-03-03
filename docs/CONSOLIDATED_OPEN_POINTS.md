# Consolidated Open Points — Post Rebase 2026-03-03

**Repos**: rustynum (main @ 60ab488), ladybug-rs (main @ dd750f8)
**Branch**: `claude/compare-rustynum-ndarray-5ePRn` — rebased, at main on both repos

---

## Legend

| Tag | Meaning |
|-----|---------|
| **P0** | Must fix before production / new features |
| **P1** | Fix before scale / concurrent load |
| **P2** | Fix when touching the area |
| **P3** | Nice to have / cosmetic |
| ~~struck~~ | Already resolved during or before this rebase |

---

## 1. CROSS-REPO: Duplication & Wiring (rustynum ↔ ladybug-rs)

| # | Item | Repo | Status |
|---|------|------|--------|
| X1 | ~~Delete `ladybug-rs/src/core/simd.rs`~~ | ladybug-rs | **DONE** — file deleted, all call sites use `rustynum_accel` |
| X2 | ~~Wire `rustynum_accel` into core search~~ | ladybug-rs | **DONE** 2026-02-27 |
| X3 | ~~Fix `.to_vec()` in `rustynum_accel.rs:148`~~ | ladybug-rs | **DONE** 2026-02-27 |
| X4 | Fix scalar Hamming loop in `bind_space.rs:1848-1855` | ladybug-rs | **P1** — use `rustynum_accel::slice_hamming` |
| X5 | Wire `CogRecordView<'a>` into `hydrate_nodes()` for zero-copy Lance→BindSpace | ladybug-rs | **P1** — view exists in rustynum-arrow |
| X6 | Wire `BindBridge` into ladybug server startup for zero-serde awareness hydration | ladybug-rs | **P1** — N1 in ladybug CLAUDE.md |
| X7 | Wire `rustynum-holo` Overlay/DeltaLayer/LayerStack into BindSpace | ladybug-rs | **P2** |
| X8 | Wire `rustynum-clam` ClamTree for indexed sub-linear search | ladybug-rs | **P2** |
| X9 | Wire `rustynum-holo` LodIndex/lod_knn_search for LOD pruning | ladybug-rs | **P2** — blocked by X8 |
| X10 | Expose `handle_request_body` from crewai-rust server | crewai-rust | **P1** — blocks ladybug integration |
| X11 | Expose `handle_api_post/get` from n8n-grpc | n8n-rs | **P1** — blocks ladybug integration |

---

## 2. RUSTYNUM — Correctness & Safety

### P0 — Zero-Copy Breaks

| # | Item | Location |
|---|------|----------|
| R1 | Migrate callers to `CogRecordView<'a>` — 4× `.to_vec()` in `record_batch_to_cogrecords()` | `rustynum-arrow/src/arrow_bridge.rs:126-159` |
| R2 | Migrate callers to `CascadeIndices::build_from_arrow()` — 4× `extend_from_slice()` = 819 MB alloc | `rustynum-arrow/src/indexed_cascade.rs:83-111` |

### P0 — Remaining Panic Sites (26 sites → need `try_*` / Result)

| # | File | Sites |
|---|------|-------|
| R3 | `array_struct.rs` | 9: new_with_shape, item, dot, min/max_axis, log, argmin, argmax, top_k |
| R4 | `statistics.rs` | 5: mean_axis, sum_axis, var_axis, percentile, percentile_axis |
| R5 | `bitwise.rs` | 6: bitand, bitxor, bitor shape checks |
| R6 | `manipulation.rs` | 3: flip_axis, squeeze |
| R7 | `operations.rs` | 1: remaining panic |

### Green (fix when touching file)

| # | Item | Location |
|---|------|----------|
| R8 | N16 — Hardcoded concept τ addresses (0x40–0xE0) | `ghost_discovery.rs` |
| R9 | N23 — LFD integer division rounding | `clam/tree.rs` |
| R10 | N26 — 4 `.expect()` chains on schema columns | `datafusion_bridge.rs` |
| R11 | N28 — I32/I64 DType stubs (allocated but no typed getter) | `blackboard.rs` |
| R12 | N30 — `pruned_subtrees` always zero | `compress.rs:305` |
| R13 | U4 — `learn_improves` asserts >= 0 on usize (tautology) | `recognize.rs:1113` |

### P2 — Code Quality

| # | Item | Effort |
|---|------|--------|
| R14 | Macro-dedup Python bindings: `array_f32.rs`/`array_f64.rs` (~600→150 LOC) | 1 hour |
| R15 | Extract carrier/focus/phase to rustynum-common (5,343 dedup lines across 4 archive crates) | 3 hours |
| R16 | Fix `qualia_xor` crate — missing candle_core, candle_nn, tokenizers deps (7 errors) | Small |
| R17 | `select_hamming_fn()` wiring to replace simd.rs (P0 TODO in CROSS_PLANE_TECHNICAL_MAPPING) | Medium |

### Oracle Performance (rustynum-oracle/TODO.md)

| # | Item | Current | Target |
|---|------|---------|--------|
| R18 | Single projection (64K bits) | 56 ms | < 10 ms (needs SIMD) |
| R19 | Batch projection (100 × 64K) | 5.6 s | < 500 ms (needs SIMD) |

---

## 3. LADYBUG-RS — Storage Race Conditions

### P0 — Fix Before Production (Data Loss)

| # | Item | Location | Issue |
|---|------|----------|-------|
| L1 | **WAL is write-behind** — writes memory first, then disk; crash = data loss | `storage/hardening.rs` | WAL.append() + sync() must come BEFORE BindSpace.write_at() |
| L2 | **Temporal serializable conflict gap** — window between check_conflicts() and commit() | `storage/temporal.rs` | Acquire WRITE lock for entire commit |
| L3 | **XorDag parity TOCTOU** — parity computed after releasing bind_space lock | `storage/xor_dag.rs` | Hold lock through entire commit + parity |
| L4 | **LRU duplicate entries** — touch() drops lock between tracker and queue | `storage/hardening.rs` | Hold both locks atomically |
| L5 | **FINGERPRINT_WORDS (156) vs FINGERPRINT_U64 (157) mismatch** | `bind_space.rs:53` vs `lib.rs` | Single source of truth — pick one |

### P1 — Fix Before Scale

| # | Item | Location |
|---|------|----------|
| L6 | WriteBuffer ID allocation gap — flusher may miss writes | `storage/resilient.rs` |
| L7 | TieredStorage eviction race — fresh data evicted | `storage/snapshots.rs` |
| L8 | DependencyGraph partial write — two lock scopes | `storage/resilient.rs` |
| L9 | EpochGuard steal race — item pushed to drained slot | `storage/xor_dag.rs` |

### P1 — Unwrap Audit

| File | Count |
|------|-------|
| `bin/server.rs` | 70 |
| `storage/substrate.rs` | 47 |
| `storage/lance_persistence.rs` | 46 |
| `storage/snapshots.rs` | 45 |
| `storage/service.rs` | 44 |
| **Total across codebase** | **~400+** |

---

## 4. LADYBUG-RS — Lance / Persistence

| # | Item | Priority |
|---|------|----------|
| L10 | **Lance API mismatch** — Cargo.toml says `lance = "1.0"` but vendor is 2.1.0-beta.0 | **P2** (blocks S3) |
| L11 | Implement `lance-io::ObjectStore` + scheduler → `persistence/io.rs` | **P0** (zero persistence today) |
| L12 | Implement `lance::Dataset::write()` + InsertBuilder → `persistence/writer.rs` | **P0** |
| L13 | Implement `lance::dataset::fragment` → `persistence/fragment.rs` | **P0** |
| L14 | Implement commit/Manifest → `persistence/manifest.rs` | **P1** |
| L15 | Configure encoding (Bitpacking for structure, Plain for fingerprint) | **P1** |
| L16 | Implement Session → LRU page cache | **P1** |
| L17 | No S3 backup implementation (pseudocode only) | **P2** |

---

## 5. LADYBUG-RS — Orchestration & Flight

| # | Item | Priority |
|---|------|----------|
| L18 | Hierarchical dispatch = sequential (not differentiated) | **P3** |
| L19 | Task dependency resolution not enforced (depends_on field ignored) | **P3** |
| L20 | A2A message delivery status bug (send marks Delivered, receive filters Pending) | **P3** |
| L21 | Flight server has placeholder action implementations | **P2** |
| L22 | `#![allow(dead_code)]` crate-wide in `lib.rs:56` | **P3** |

---

## 6. LADYBUG-RS — Test Failures (experimental features)

10 failures under experimental feature flags — not blocking default build:

| Test | Root Cause | Effort |
|------|-----------|--------|
| `collapse_gate::test_sd_calculation` | Algorithm threshold logic | Medium |
| `quantum_ops::test_permute_adjoint` | Permute not inverse | Medium |
| `cypher::test_variable_length` | ParseFloatError in tokenizer | Small |
| `causal_ops::test_store_query_correlation` | Empty query result | Small |
| `context_crystal::test_temporal_flow` | Insert not persisting (popcount=0) | Medium |
| `nsm_substrate::test_codebook_initialization` | primes < 60 | Small |
| `nsm_substrate::test_learning` | vocab < 65 | Small |
| `jina_cache::test_cache_hit_rate` | Off-by-one | Trivial |
| `crystal_lm::test_serialize` | None unwrap | Small |
| `causal::test_correlation_store` | Empty query result | Small |

---

## 7. PHASE 1 REMEDIATION — Substrate Architecture (ladybug-rs)

From `PHASE1_REMEDIATION.md` — structural refactoring of storage layer:

| Area | Open Tasks | Priority |
|------|-----------|----------|
| Substrate.rs cleanup (delete extended_nodes/edges, add HdrCascade) | 6 tasks | P1 |
| Search with HDR Cascade (replace O(32K) linear scan) | 5 tasks | P1 |
| Fluid zone lifecycle (tick, TTL, demotion) | 5 tasks | P2 |
| CogRedis cleanup (unused imports, fps) | 2 tasks | P3 |
| CAM operations routing (BIND, RESONATE, HAMMING) | 4 tasks | P2 |
| Crystallization / evaporation | 3 tasks | P2 |

---

## Summary by Priority

| Priority | rustynum | ladybug-rs | Cross-repo | Total |
|----------|----------|------------|------------|-------|
| **P0** | 2 (zero-copy) + 26 panic sites | 5 (race conditions) + 3 (persistence) | 0 | **36** |
| **P1** | 0 | 4 (races) + 3 (persistence) + ~400 unwraps + 16 substrate tasks | 4 (wiring) | **~427** |
| **P2** | 4 (quality) + 2 (perf) | 6 (lance/flight/substrate) | 3 (holo/clam) | **15** |
| **P3** | 6 (green/cosmetic) | 4 (orch) + 2 (cleanup) + 10 (test fixes) | 0 | **22** |

### Recommended Next Actions (highest impact)

1. **L1–L5**: Fix the 5 storage race conditions — they risk data loss / corruption
2. **R1–R2**: Wire zero-copy Arrow paths — eliminates ~1 GB unnecessary allocation
3. **R3–R7**: Convert 26 remaining panics to Result types (try_* variants pattern already established)
4. **L5**: Resolve FINGERPRINT_WORDS vs FINGERPRINT_U64 — single constant, one source of truth
5. **X6**: Wire BindBridge into server startup — unblocks zero-serde hydration pipeline

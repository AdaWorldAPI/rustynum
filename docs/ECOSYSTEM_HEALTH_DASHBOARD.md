# Ecosystem Health Dashboard

**Generated**: 2026-03-03

---

## Build Status

| Metric | rustynum | ladybug-rs | n8n-rs | neo4j-rs | crewai-rust | aiwar-neo4j-harvest |
|---|---|---|---|---|---|---|
| **Compiles** | YES | NO (crewai dep) | YES (needs protoc) | NO (transitive crewai) | Not in workspace | YES |
| **Tests** | 57/57 pass | Blocked | 134/134 pass | Blocked | -- | -- |
| **Warnings** | 0 | Unknown | 1 (unused import) | Unknown | -- | -- |
| **Clippy** | Clean | Blocked | Clean | Blocked | -- | -- |
| **.unwrap()** | 103 | 966 | ~114 (~110 in tests) | 110 | -- | -- |
| **TODOs** | 8 | 16 | 1 | 1 | -- | -- |
| **Rust LOC** | ~95,713 | ~168,165 | ~28,258 | ~14,819 | ~73,657 | ~590 |

**Healthy repos**: rustynum (57/57), n8n-rs (134/134)
**Blocked repos**: ladybug-rs and neo4j-rs (same root cause)
**Not in workspace**: crewai-rust

---

## The Single Blocker: crewai-rust

Three of four main repos are broken by the same root cause:

```
ladybug-rs/Cargo.toml  ->  crewai-rust (default feature, OBLIGATORY path dep)
neo4j-rs               ->  ladybug-rs  ->  crewai-rust (transitive)
n8n-rs                 ->  ladybug-rs  ->  crewai-rust (transitive)
```

crewai-rust is not provisioned in this workspace, and even `optional = true` path deps are resolved at manifest load time.

**Fix**: Make `crewai` a non-default feature in ladybug-rs. Move `"crewai"` from `default = [...]` to opt-in. This single change unblocks compilation of ladybug-rs, neo4j-rs, and n8n-rs.

n8n-rs has a second environment blocker: `protoc` (protobuf compiler) needed by n8n-grpc build script.

---

## Technical Debt Ledger

### Critical (blocks all other work)

1. **Make crewai a non-default feature in ladybug-rs** -- Unblocks 3 repos in one shot
2. **Install protoc or pre-generate n8n-grpc** -- `apt-get install protobuf-compiler` or commit generated `.rs`

### High (production panic risk)

1. **ladybug-rs: 966 .unwrap() calls** -- Worst offenders: server.rs, lance.rs, lance_v1.rs, Arrow array construction. Each is a potential production crash.
2. **ladybug-rs: 9 documented race conditions** -- 2 CRITICAL/HIGH severity in WAL and XOR DAG. Need `parking_lot::RwLock` or channel-based designs.
3. **ladybug-rs: 10 known test failures** -- cypher::test_variable_length (ParseFloatError) blocks lance-graph Cypher integration.

### Medium (code quality)

1. **ladybug-rs: 3 empty benchmark files** -- benches/cam_ops.rs, fingerprint.rs, hamming.rs are 9-line stubs
2. **ladybug-rs: Delete src/core/simd.rs** -- 348 lines duplicating rustynum logic (marked N2 in CLAUDE.md, "small effort")
3. **neo4j-rs: Cost-based optimizer** -- Single TODO in planner. Not urgent.
4. **n8n-rs: tokio = { features = ["full"] }** -- Pull only needed features to reduce compile time

### Low (nice-to-have)

1. **rustynum: SIMD acceleration for var/transpose/dot** -- 5 of 7 TODOs are scalar fallback paths
2. **Unified error type across repos** -- All four repos have different error enums. Shared `ada-error` crate would help.
3. **rustynum: Result-based error handling** -- 103 .unwrap() calls. operations.rs TODO says "Return Result instead of panicking"

---

## Open Integration Work

### lance-graph Query Pipeline into ladybug-rs

**Status**: ladybug-rs has `src/query/cypher.rs` that transpiles Cypher->SQL as text. lance-graph has proper AST -> LogicalPlan -> DataFusion pipeline.

**Action**: Replace ladybug's string-based Cypher->SQL with lance-graph's `CypherQuery::execute()` which goes through typed LogicalOperator nodes. Gives graph-aware optimization, parameterized queries, variable-length path support for free.

**Integration point**: `src/query/datafusion.rs:131` (TODO: "read all data into memory -- should use Lance TableProvider")

### GraphSourceCatalog as Universal Storage Abstraction

**Status**: lance-graph's `GraphSourceCatalog` trait cleanly separates node/relationship resolution from storage. ladybug-rs has 3 competing storage paths. neo4j-rs has `StorageBackend` trait (good) but doesn't compose with ladybug's paths.

**Action**: Shared `StorageCatalog` trait in ladybug-contract that lance-graph's `GraphSourceCatalog`, neo4j-rs's `StorageBackend`, and ladybug's BindSpace all implement.

### n8n-rs: Complete the Arrow Flight Chain

**Status**: Workflow engine, expression evaluator, node types exist. Arrow Flight chain (n8n -> Arrow -> ladybug -> rustynum) not fully wired.

**Action**: Wire `n8n-contract/ladybug_router.rs` to call ladybug's BindSpace via Arrow Flight. Currently routing stubs. Blocked by ladybug compile.

---

## Recommendation: Where to Start

| Goal | Action | Impact |
|---|---|---|
| Maximum unblocking | Fix debt #1-2 (crewai optional, install protoc) | Lights up 3 repos |
| Highest-value expansion | lance-graph query pipeline into ladybug-rs | Replaces weakest part with strongest |
| Safest production impact | rustynum Result-based errors | Only fully green repo, zero regression risk |

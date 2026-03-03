# Consolidated Open Points — All Repos

**Updated**: 2026-03-03 | Compiled from all session audits

---

## RUSTYNUM (~94K LOC) — 57/57 tests, 0 warnings

### Resolved
- [x] GEMM bounds check (assert at int8_gemm.rs:285)
- [x] qualia_xor deps feature-gated behind `bert`
- [x] rustymkl VML: vsexp() + 37 functions implemented and tested
- [x] portable_simd → std::arch port (870 occurrences)
- [x] QualiaGateLevel + CollapseGate (47 tests)
- [x] `try_div_broadcast()` added — last remaining `panic!` in Div trait now has fallible variant

### Partial (verified 2026-03-03)
- [ ] **34 public API panics → try_* variants**: 28 try_* variants exist (including new `try_div_broadcast`). Panicking wrappers documented with `# Panics` but still panic on Err. ~56 non-test panic!/assert!/.unwrap() remain in rustynum-rs/src/num_array/. Std::ops traits can't return Result — this matches NumPy behavior. Status: PARTIAL, not resolved.
- [ ] **Zero-copy CogRecordView migration**: `cogrecord_views()` exists, `record_batch_to_cogrecords()` deprecated. **1 production caller** remains: `lance_io.rs:53`. 7 test callers use deprecated path.
- [ ] **Zero-copy CascadeIndices migration**: `build_from_arrow()` exists, `build()` deprecated. 6 test callers use copying path. 0 production callers outside tests.
- [ ] **Gate enforcement wire**: QualiaGateLevel implemented, enforcement correctly lives in crewai-rust.

### Open
- [ ] **Wire rustyblas GEMM → horizontal_sweep**: `sgemm()` exists, not wired into `horizontal_sweep.rs`
- [ ] **Wire jitson JIT → horizontal_sweep**: `JitEngine::compile_scan()` exists, not wired
- [ ] **Python binding f32/f64 dedup**: `array_f32.rs` (185 lines) vs `array_f64.rs` (213 lines). PyO3 limitation, low priority.
- [ ] **Qualia corpus expansion 231 → 1024**: 231 items in 44 families. No expansion started.
- [ ] **Nib4 BF16 hydration**: No changes found.

### Quick Win
Migrate `lance_io.rs:53` to `cogrecord_views()` + update 6 test callers to `build_from_arrow()`. Mechanical refactor.

---

## LADYBUG-RS (~134K LOC) — 191 clippy warnings, 10 test failures in experimental features

### Resolved
- [x] All 5 rustynum crates wired into BindSpace (hamming, phase, carrier, focus, clam)
- [x] BindSpace-native rustynum integration complete
- [x] Age-based hot→cold flush to Lance
- [x] Arc<Mutex> eliminated (split_at_mut / parallel_into_slices)
- [x] Overlay zero-copy bridge (as_fingerprint_words())

### Open — P0
- [ ] **N2: Consolidate src/core/simd.rs** — 348 lines overlapping with rustynum SIMD. **KEEP as AVX2/CPU silent fallback** for machines without AVX-512. Only deduplicate ops that rustynum-core already covers with its own scalar fallback path. Medium effort.
- [ ] **N1: Wire BindBridge into server startup** — zero-serde awareness hydration. Medium effort. Cross-repo with crewai-rust.
- [ ] **N3: Wire CogRecordView into hydrate_nodes()** — zero-copy Lance → BindSpace. Medium effort.
- [ ] **N9: Add periodic flush_aged() timer** — every 5-30 min in server.rs. Small effort.
- [ ] **CR-P0: Implement SubstrateView for BindSpace** — crewai-rust bridge, removes HTTP hop. Medium effort.

### Open — P1
- [ ] **N4: Expose handle_request_body from crewai-rust** (vendor-crewai). Not blocked.
- [ ] **N5: Expose handle_api_post/get from n8n-grpc** (vendor-n8n). Not blocked.
- [ ] **N6: Add rustynum-holo Overlay/DeltaLayer/LayerStack to BindSpace**. Not blocked.
- [ ] **N7: Wire ClamTree::build() for indexed sub-linear search**. Not blocked.
- [ ] **Lance API mismatch**: Cargo.toml says lance="1.0" but vendor has 2.1.0. FixedSizeList hydration (`lance.rs:714`) — embeddings always None.

### Open — P2
- [ ] **N8: Wire LodIndex/lod_knn_search()** — depends on N7
- [ ] **966 .unwrap() calls** — worst offenders: server.rs, lance.rs, lance_v1.rs
- [ ] **2 CRITICAL race conditions** — WAL and XOR DAG
- [ ] **3 empty benchmark stubs** — benches/cam_ops.rs, fingerprint.rs, hamming.rs
- [ ] **10 test failures** — collapse_gate, quantum_ops, cypher, causal_ops (experimental features)

### Open — Infra
- [ ] **191 clippy warnings** — mostly unused imports in vendored crewai-rust code + Default::default() patterns

---

## CREWAI-RUST (~74K LOC) — 670 tests, 0 warnings (standalone)

### Resolved
- [x] Core executor wired
- [x] JITSON integration steps J.1–J.6 complete
- [x] ThinkingStyle → 36 styles in 6 clusters, 23D cognitive space

### Open — Tier 1 (E2E Agent Execution)
- [ ] **Wire real tool executor**: `core.rs:541` — callback returns fake results. ~50 LOC. **Single highest-leverage item.**
- [ ] **execute_with_timeout()**: `core.rs:456` — 5 LOC, wrap invoke in `tokio::time::timeout`
- [ ] **E2E integration test**: `crew.kickoff()` → LLM → tool → real output

### Open — Tier 2 (RAG Pipeline)
- [ ] **OpenAI embeddings provider**: All 12 embedding providers are stubs. OpenAI first unlocks RAG.
- [ ] **TextLoader + CsvLoader**: `loaders.rs` — all `bail!()`. 10-20 LOC each.
- [ ] **Default chunker**: Paragraph/fixed-size split. ~20 LOC.

### Open — Tier 3 (Protocol)
- [ ] **MCP JSON-RPC client**: `client.rs` — transport works, `call_tool()` returns error.
- [ ] **A2A client HTTP**: `client.rs` — `send_message()` / `get_agent_card()` all `bail!()`.
- [ ] **LiteLLM bridge**: Fallback to Ollama/Groq/Together. Low priority (3 providers work).

### Open — Cross-repo
- [ ] **CR-P0b: Wire JitProfile into ModuleRuntime activation** — τ addresses → jitson compile. Not blocked.
- [ ] **N10: BERT embedding for blood-brain barrier inbound** — external model needed.

---

## NEO4J-RS (~10-15K LOC) — 177 tests, 0 warnings

### Resolved
- [x] UNWIND parser + planner
- [x] MATCH...MERGE piped execution
- [x] 177 tests passing, 0 ignored

### Open
- [ ] **WITH clause planner/executor**: Parser handles it, planner doesn't generate plans. ~150 lines. **Main Cypher gap.**
- [ ] **42 parser panics → Result**: `parser.rs` panics on user input. Medium effort.
- [ ] **Variable-length path execution**: Medium effort.
- [ ] **Cost-based optimizer**: Large, future. Rule-based works for medium queries.
- [ ] **LadybugBackend testing**: Needs ladybug-rs + protoc to build with `ladybug` feature.

---

## N8N-RS (~28K LOC) — 134/134 tests (when protoc available)

### Resolved
- [x] JITSON integration (all 6 steps)
- [x] Arrow 57 / DataFusion 51 aligned

### Open
- [ ] **N8N-P0: CompiledStyleRegistry → jitson compile end-to-end**. Not blocked.
- [ ] **N8N-P0b: In-process delegation** — replace HTTP proxy with vendor-linked calls. Depends on N4, N5.
- [ ] **N8N-P1: Arrow Flight endpoint for rustynum kernels** (port 50052). Not blocked.
- [ ] **Flight streaming**: `n8n-arrow/src/flight.rs` — `do_exchange` unimplemented.
- [ ] **1 TODO** in codebase.

### Infra
- [ ] **protoc not installed** — blocks full build of n8n-grpc. `apt-get install protobuf-compiler`.

---

## OPENCLAW-RS (~LOC unknown)

### Open
- [ ] **Real AgentRuntime (Anthropic API)**: Currently stub.
- [ ] **Streaming chat deltas**: Stub word-split.
- [ ] **INDEX container (#7)**: Unlocked by rustynum `SynapseState` + `crystallize::<N>()`. Can implement now.
- [ ] **EMBED container (#8)**: Unlocked by `soaking.rs` int8 bridge. Can implement now.
- [ ] **Channel adapters**: Web, n8n bridge, crewai integration.
- [ ] **Memory backend hardcoded to InMemory**: Config flag selection needed.
- [ ] **6 warnings**: ws.rs, hook, memory, config.
- [ ] **No tests, no Docker/CI**.

---

## AIWAR-NEO4J-HARVEST (~590 LOC)

No open points. Functional CLI for Cypher generation, Neo4j ingest, chess knowledge.

---

## Cross-Repo Priority Matrix

| Priority | Item | Repos | Effort | Impact |
|---|---|---|---|---|
| **P0** | Wire real tool executor | crewai-rust | Small | Agents can think but can't act |
| **P1** | Consolidate simd.rs (N2) | ladybug-rs | Medium | Dedupe only where rustynum-core has scalar fallback; keep AVX2 path |
| **P0** | SubstrateView impl (CR-P0) | ladybug-rs + crewai-rust | Medium | Removes HTTP hop |
| **P0** | Install protoc | infra | Trivial | Unblocks n8n-rs + ladybug-rs flight |
| **P1** | WITH clause planner | neo4j-rs | Medium | Main Cypher feature gap |
| **P1** | OpenAI embeddings provider | crewai-rust | Medium | Unlocks entire RAG pipeline |
| **P1** | Wire GEMM + jitson → horizontal_sweep | rustynum | Medium | Performance |
| **P1** | Zero-copy migrations (lance_io + tests) | rustynum | Small | Eliminate allocations |
| **P1** | INDEX + EMBED containers | openclaw-rs | Medium | Unlocked by rustynum |
| **P2** | 966 unwraps in ladybug-rs | ladybug-rs | Large | Production stability |
| **P2** | 42 parser panics in neo4j-rs | neo4j-rs | Medium | User input safety |
| **P2** | In-process n8n delegation | n8n-rs + ladybug-rs | Medium | Removes HTTP proxy |
| **P2** | Clippy cleanup (191 warnings) | ladybug-rs | Medium | Code quality |

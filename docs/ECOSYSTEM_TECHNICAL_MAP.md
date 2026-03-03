# Ada Rust Ecosystem -- Technical Map

**Generated**: 2026-03-03 | **Total**: ~340,000 lines of Rust | **Repos**: 6 | **Toolchain**: stable Rust 1.93

---

## Ecosystem Overview

| Repository | LOC | Role | Level |
|---|---|---|---|
| **rustynum** | ~93,726 | SIMD compute, bindspace types, BLAS/MKL/JIT | L1 -- Surface (spatial substrate) |
| **ladybug-rs** | ~134,352 | Cognitive database, BindSpace, storage, Arrow/Lance | L2 -- Awareness (temporal process) |
| **crewai-rust** | ~73,657 | Agent orchestration, Blackboard, drivers, personas | L4 -- Composition (behavioral) |
| **n8n-rs** | ~28,258 | Workflow engine, multi-transport, Arrow Flight | Meta -- workflow transcode |
| **neo4j-rs** | ~9,883 | Property graph, Cypher parser, StorageBackend | L3 -- Reasoning (structural) |
| **aiwar-neo4j-harvest** | ~590 | Data harvester for AI War Cloud + chess | Data ingestion tooling |

### Dependency Direction (LAW -- never violated)

```
rustynum-core  (types + SIMD)
    ^
    |
rustynum-holo  (holographic containers)
    ^
    |
ladybug-rs     (BindSpace, storage)
    ^
    |
crewai-rust    (Blackboard, agents)
```

n8n-rs bridges all levels. neo4j-rs plugs into ladybug-rs via LadybugBackend.

---

## 1. RUSTYNUM -- The Hardware Layer

### 1.1 Workspace Crates

| Crate | Path | Purpose |
|---|---|---|
| `rustynum-core` | `rustynum-core/` | SIMD dispatch, BF16 distance, K0/K1/K2 kernels, Fingerprint, DeltaLayer, LayerStack, CollapseGate, Blackboard (arena), CPUID detection |
| `rustynum-rs` | `rustynum-rs/` | NumArray (f32/f64/u8), HDC operations (bundle, bind, hamming), CogRecord bridge |
| `rustynum-holo` | `rustynum-holo/` | Holographic containers: Overlay, MultiOverlay, AlignedBuf2K, phase-space, carrier model, Gabor wavelets, LOD pyramid, focus masks, delta layers |
| `rustynum-arrow` | `rustynum-arrow/` | Arrow/Lance bridge: CogRecordView, CascadeIndices, horizontal sweep, DataFusion cascade scan |
| `rustynum-clam` | `rustynum-clam/` | CLAM tree, triangle-inequality search, panCAKES compression, QualiaCAM |
| `rustynum-bnn` | `rustynum-bnn/` | Binary Neural Network inference, K0/K1 Belichtungsmesser cascade, RIF-Net, cross-plane partial binding, causal trajectory |
| `rustyblas` | `rustyblas/` | Full BLAS Level 1/2/3: GEMM (f32, f64, BF16, INT8), quantization |
| `rustymkl` | `rustymkl/` | Pure Rust MKL replacement: LAPACK (LU, Cholesky, QR, SVD), FFT, VML |
| `rustynum-oracle` | `rustynum-oracle/` | Sweetspot evaluation, recognition, ghost discovery, NARS, sweep, organic heuristics |
| `qualia_xor` | `qualia_xor/` | Qualia corpus experiments: BERT embedding comparison, edge-vectors, cypher-vsa, hydrate-agents |
| `jitson` | `jitson/` | **Separate workspace** -- Cranelift JIT: JSON/YAML config to native function pointers (AVX-512 patched wasmtime fork) |

**Archive crates** (sacred -- do not modify): `rustynum-archive/`, `rustynum-archive-v3/`, `rustynum-carrier/`, `rustynum-focus/`

### 1.2 Public API: Core Types

#### Fingerprint\<N\>

```rust
pub struct Fingerprint<const N: usize> {
    words: [u64; N],
}
```

THE universal container. `Fingerprint<256>` = 2048 bytes = `[u64; 256]`.

- Operations: `xor()`, `and_popcount()`, `popcount()`, `hamming_distance()`, `similarity()`, `is_empty()`
- Wire: `as_bytes()` / `from_bytes()`
- **Zero-copy identity**: same memory as `Overlay.buffer` via `as_fingerprint_words()` pointer reinterpret

#### DeltaLayer\<N\>

```rust
pub struct DeltaLayer<const N: usize> {
    delta: Fingerprint<N>,
    writer_id: u32,
}
```

XOR delta from ground truth. Writer owns `&mut` exclusively.
- `write(&ground, &desired)` -- compute XOR delta
- `read(&ground) -> Fingerprint<N>` -- ground XOR delta = view
- `conflict(&other) -> u32` -- AND + popcount measures contradiction

#### LayerStack\<N\>

Multi-writer concurrent state via XOR delta layers over immutable ground truth.
- `evaluate(threshold) -> CollapseGate` -- SD-based decision
- `commit()` -- XOR all deltas into ground truth (exclusive `&mut self`)

#### CollapseGate

```rust
pub enum CollapseGate { Flow, Hold, Block }
```

- `Flow` (SD < 0.15): clear winner, collapse
- `Hold` (0.15-0.35): maintain superposition
- `Block` (> 0.35): cannot collapse, ask for clarification

#### Blackboard (arena)

```rust
pub struct Blackboard {
    buffers: HashMap<String, BufferMeta>,
}
```

64-byte aligned allocations (AVX-512 cache-line). Split-borrow API: `borrow_3_mut_f32("A", "B", "C")` gives 3 `&mut [f32]` simultaneously. Sound because each buffer is a separate heap allocation. `!Send + !Sync`.

### 1.3 K0/K1/K2 Kernel Pipeline

Two fixed SKUs: **SKU-16K** (16384 bits/256 words/2048 bytes) and **SKU-64K** (65536 bits/1024 words/8192 bytes).

```rust
pub fn k0_probe(a: &[u64], b: &[u64]) -> u32;          // 64-bit XOR+POPCNT, eliminates ~55%
pub fn k1_stats(a: &[u64], b: &[u64]) -> u32;          // 512-bit (8 words), eliminates ~90% of survivors
pub fn k2_exact(a: &[u64], b: &[u64]) -> EnergyConflict; // Full width

pub struct EnergyConflict { conflict: u32, energy_a: u32, energy_b: u32 }
pub struct SliceGate { k0_reject: u32, k1_reject: u32, k2_hot/mid/cold/anti: u32 }
```

Sigma-significance scoring: `SigmaScore`, `score_sigma()`, per-word histogram via `K2Histogram`.
HDR scoring: 0-6 scale (HDR_ANTI through HDR_BLAZING).

### 1.4 Hybrid Pipeline

```rust
pub struct HybridScore {
    hamming_distance: u32, hdr: HdrScore, energy: EnergyConflict,
    sigma: SigmaScore, bf16_distance: u64, structural_diff: BF16StructuralDiff, ...
}
```

Bridges K0/K1/K2 + BF16 structured distance + awareness substrate.
- Tier 0 prefilter: VNNI INT8 dot-product (VPDPBUSD) -- 64 bytes/cycle
- BF16 tail: sign/exp/man weighted distance on survivors
- Awareness states: Crystallized (00), Tensioned (01), Uncertain (10), Noise (11)

### 1.5 BF16 Structured Distance

```rust
pub struct BF16Weights { sign: u16, exponent: u16, mantissa: u16 }
pub const JINA_WEIGHTS: BF16Weights = { sign: 256, exponent: 32, mantissa: 1 };
pub const TRAINING_WEIGHTS: BF16Weights = { sign: 1024, exponent: 64, mantissa: 0 };
```

Per-BF16-field weighting: sign flip = class-level change, exponent = attention, mantissa = noise.
Awareness decomposition: 2-bit per-dimension classification packed into `Vec<u8>`.

### 1.6 Spatial Resonance

3D BF16 crystal for SPO encoding: X=Subject, Y=Predicate, Z=Object.
- XOR bind: `X = S xor P`, `Y = P xor O`, `Z = S xor O`
- Recovery: `S = X xor P`, `O = Y xor P` (XOR self-inverse)

### 1.7 SIMD Architecture

All on **stable Rust 1.93** -- zero nightly features.

- `simd_compat.rs` wraps `std::arch::x86_64` intrinsics with portable_simd-compatible API
- Types: `F32x16`, `F64x8`, `U8x64`, `I32x16`, `I64x8`, `U32x16`, `U64x8` (all `#[repr(transparent)]` over `__m512*`)
- Runtime CPUID detection: `select_hamming_fn()`, `select_dot_i8_fn()`, `select_bf16_hamming_fn()` cached in `OnceLock`
- Compute dispatch hierarchy: INT8 prefilter -> VNNI INT8 GEMM -> BF16 GEMM -> FP32 GEMM -> GPU offload

### 1.8 BLAS (rustyblas)

**Level 1** (vector-vector, 16 ops): sdot/ddot, saxpy/daxpy, sscal/dscal, snrm2/dnrm2, sasum/dasum, isamax/idamax, scopy/dcopy, sswap/dswap

**Level 2** (matrix-vector, 10 ops): sgemv/dgemv, sger/dger, ssymv/dsymv, strmv/dtrmv, strsv/dtrsv

**Level 3** (matrix-matrix, 8 ops): sgemm/dgemm, ssymm/dsymm, strsm/dtrsm, ssyrk/dsyrk

- Goto microkernel: MR=6 x NR=16 (f32), MR=6 x NR=8 (f64)
- Cache blocking: KC=256 (L1), MC=128 (L2), NC=1024 (L3)
- **138 GFLOPS** at 1024x1024 on 16 threads

**Quantized GEMM**: INT8 (VNNI VPDPBUSD), BF16 (FP32 accumulate), INT4 quantize/dequantize.

### 1.9 MKL Replacement (rustymkl)

- **LAPACK**: LU factorization, Cholesky, QR, eigenvalues, SVD
- **FFT**: Radix-2 Cooley-Tukey (in-place), split-radix, complex/real (f32/f64). SIMD butterfly via `f32x16`/`f64x8`.
- **VML**: `vsexp()`, `vsln()`, `vssin()`, `vscos()`, `vssqrt()`, `vspow()` (vectorized math)

### 1.10 CLAM Integration (rustynum-clam)

- **ClamTree**: Divisive hierarchical clustering, LFD estimation
- **Distance trait**: Generic -- plug any metric (Hamming, cosine, edit distance)
- **Triangle-inequality search**: Exact k-NN/rho-NN with d_min/d_max pruning (CAKES DFS Sieve)
- **panCAKES compression**: Hierarchical XOR-diff from cluster centers (5-70x compression)
- **QualiaCAM**: CLAM + qualia, `ClamPath` B-tree keys, `CollapseGateBias`
- **Semantic protocol**: `parse_command()` / `command_to_query()` -- natural language to DataFusion

### 1.11 BNN Layer (rustynum-bnn)

- K0/K1 Belichtungsmesser (exposure meter) cascade
- Progressive sampling: K0 (64-bit, ~84% reject) -> K1 (512-bit, ~97.5%) -> BF16 (~99.7%) -> Full
- RIF-Net (Zhang et al. 2025): `BPReLU`, `BinaryBatchNorm`, `RifCaBlock`, `RifFlowMetrics`
- Cross-Plane Partial Binding: 6 halo types, SPO inference, lattice climber
- Causal Trajectory: NARS x Fovea x Context resonator

### 1.12 Holographic Layer (rustynum-holo)

**Core**: `Overlay` IS `Fingerprint<256>` via `as_fingerprint_words()` (zero-copy).

**Phase-space**: `phase_bind_i8()`, `phase_unbind_i8()`, `wasserstein_sorted_i8()`, `circular_distance_i8()`, `phase_histogram_16()`, `phase_bundle_circular()`, `project_5d_to_phase()` / `recover_5d_from_phase()`.

**Carrier model**: `carrier_encode()` / `carrier_decode()` -- frequency-domain concept encoding. `carrier_bundle()` -- 32 VPADDB vs ~500 trig ops. `spectral_distance()`.

**Focus system**: `focus_xor()`, `focus_read()`, `focus_hamming()`, `focus_l1()`, `FocusRegistry`, `CompactDelta`.

**LOD Pyramid**: `lod_knn_search()` -- hierarchical pruning for sub-linear search.

**Gabor wavelets**: `gabor_read()` / `gabor_write()` -- spatial frequency analysis. Container lifecycle: Binary -> Carrier -> Gabor.

### 1.13 jitson -- Cranelift JIT

```rust
pub struct JitEngine { ... }
pub struct ScanParams { threshold: u32, top_k: usize, prefetch_ahead: usize, focus_mask: Option<Vec<u16>> }
pub struct ScanKernel { fn_ptr: *const u8 }
```

JSON/YAML config values become Cranelift immediates: `threshold: 500` -> `CMP reg, 500`. Focus masks become VPANDQ bitmasks, branch weights become hints. Depends on patched wasmtime fork with full AVX-512 support. **Separate workspace** (not a member of rustynum workspace).

### 1.14 Python Bindings

Module: `_rustynum`

- Array types: `PyNumArrayF32/F64/U8`, `PyCogRecord`
- Math: `matmul`, `dot`, `norm`, `exp`, `log`, `sigmoid`, `mean`, `median`
- **HDC/VSA**: `bundle_u8`, `hamming_distance`, `hamming_batch`, `hamming_top_k`
- **INT8 GEMM**: `quantize_f32_to_u8/i8`, `int8_gemm_i32/f32`

### 1.15 Feature Flags

| Crate | Feature | Effect |
|---|---|---|
| `rustynum-core` | `avx512` (default) | AVX-512 SIMD paths |
| `rustynum-core` | `avx2` | AVX2 fallback paths |
| `rustynum-core` | `mkl` | Intel MKL FFI paths |
| `rustynum-arrow` | `arrow` (default) | Arrow interop + DataFusion cascade scan |
| `rustynum-arrow` | `datafusion` | DataFusion 51 |
| `rustynum-arrow` | `lance` | Lance dataset I/O |
| `qualia-xor` | `bert` | BERT embedding via candle |
| `rustyblas` | `avx512` / `avx2` / `mkl` | SIMD/MKL paths |

---

## 2. LADYBUG-RS -- The Cognitive Database

### 2.1 The 8+8 Address Model

16-bit address = 8-bit prefix : 8-bit slot = 65,536 total addresses.
**3-5 cycles per lookup** -- pure array indexing, no HashMap, no FPU.

| Zone | Prefix Range | Addresses | Purpose |
|---|---|---|---|
| Surface | 0x00-0x0F | 4,096 | System operations |
| Fluid | 0x10-0x7F | 28,672 | Edges, context, working memory |
| Node | 0x80-0xFF | 32,768 | Universal bind space |

Surface compartments: Lance/Kuzu (0x00), SQL/CQL (0x01), Cypher/GQL (0x02), GraphQL (0x03), NARS/ACT-R (0x04), Causal (0x05), Meta (0x06), Verbs (0x07), Concepts (0x08), Qualia (0x09), Memory (0x0A), Learning (0x0B), Agents/crewai (0x0C), Thinking Styles (0x0D), Blackboard (0x0E), A2A (0x0F).

### 2.2 Module Map

| Module | Location | Purpose |
|---|---|---|
| `bind_space.rs` | `src/storage/` | THE CORE: 8+8 addressing, O(1) arrays, all rustynum ops wired |
| `cog_redis.rs` | `src/storage/` | Redis syntax: DN.*, CAM.*, DAG.* |
| `unified_engine.rs` | `src/storage/` | Composes ACID + CSR + MVCC + ArrowZeroCopy |
| `xor_dag.rs` | `src/storage/` | ACID transactions + XOR parity |
| `lance_zero_copy/` | `src/storage/` | Pure Arrow buffers (NO lance crate), write-through |
| `lance_persistence.rs` | `src/storage/` | Lance cold-tier: flush_aged, hydrate_nodes, persist_nodes |
| `hdr_cascade.rs` | `src/search/` | HDR filtering (~7ns per candidate) |
| `collapse_gate.rs` | `src/cognitive/` | SD-based FLOW/HOLD/BLOCK |
| `layer_stack.rs` | `src/cognitive/` | 10-layer cognitive stack |
| `cam_ops.rs` | `src/learning/` | 4096 CAM operations |
| `substrate_bridge.rs` | `src/storage/` | SubstrateView impl for crewai-rust bridge |
| `cypher_bridge.rs` | `src/` | Cypher string to BindSpace operations |
| `mul/` | `src/` | Meta-Uncertainty Layer (10-layer metacognition) |
| `spo/` | `src/` | Subject-Predicate-Object core substrate |
| `grammar/` | `src/` | Grammar Triangle (universal input layer) |
| `nars/` | `src/` | Non-Axiomatic Reasoning System |
| `server.rs` | `src/bin/` | HTTP server (port 8080) |
| `flight_server.rs` | `src/bin/` | Arrow Flight gRPC (port 50051) |

### 2.3 Storage Architecture

Two-layer model:
1. **Addressing** (always int8): prefix:slot -> array index -> 3-5 cycles
2. **Compute** (adaptive): AVX-512 (~2ns), AVX2 (~4ns), scalar (~50ns)

Zero-copy chain:
```
Lance (disk, mmap) -> Arrow Buffer (64-byte aligned) -> BindSpace (O(1) lookup)
    -> rustynum-core SIMD kernels -> Result (no allocation)
```

Hot-cold tier: In-memory BindSpace with `updated_at` timestamps. `flush_aged(threshold_secs)` moves aged nodes to Lance. `hydrate_nodes()` restores from Lance.

### 2.4 BindSpace <-> Rustynum Wiring

| Category | Methods | Crate |
|---|---|---|
| SIMD Core | `hamming()`, `similarity()`, `popcount()`, `dot_i8()` | rustynum-core |
| Binary HDC | `bundle()`, `bind()`, `nearest()`, `cascade_search()` | rustynum-core |
| Phase-space | `phase_bind()`, `phase_unbind()`, `wasserstein()` | rustynum-holo |
| Carrier | `carrier_distance()`, `carrier_correlation()`, `carrier_spectrum()` | rustynum-holo |
| Focus | `focus_xor()`, `focus_read()`, `focus_hamming()`, `focus_l1()` | rustynum-holo |
| Batch search | `batch_hamming()`, `clam_top_k()` | rustynum-clam |
| Lance bridge | `lance_write_through()`, `unified_storage()`, `arrow_zero_copy()` | lance_zero_copy |

### 2.5 Vendor Dependencies

```toml
rustynum-rs = { path = "../rustynum/rustynum-rs" }
rustynum-core = { path = "../rustynum/rustynum-core", features = ["avx512"] }
rustynum-arrow = { path = "../rustynum/rustynum-arrow", features = ["arrow"] }
rustynum-holo = { path = "../rustynum/rustynum-holo", features = ["avx512"] }
rustynum-clam = { path = "../rustynum/rustynum-clam", features = ["avx512"] }
```

All **obligatory** -- not behind feature gates.

### 2.6 Vendor Features for Single Binary

```toml
vendor-n8n = ["dep:n8n-core", "dep:n8n-workflow", "dep:n8n-arrow", "dep:n8n-grpc", "dep:n8n-hamming"]
vendor-crewai = ["dep:crewai-vendor", "crewai"]
```

### 2.7 ladybug-contract

Shared substrate types in `crates/ladybug-contract/`:
- `Container` -- 16K-bit fingerprint container
- `WideMetaView` / `MetaView` -- Metadata word access
- `CogRecord` / `CogRecord8K`
- `CognitiveAddress` -- 8+8 addressing types
- `Codebook` -- CAM codebook
- `TruthValue` -- NARS truth <frequency, confidence>
- `CogPacket` -- binary wire protocol

---

## 3. CREWAI-RUST -- Agent Orchestration

1:1 Rust port of crewAI Python v1.9.3. ~74K lines.

### 3.1 Module Map

| Module | Purpose |
|---|---|
| `blackboard/view.rs` | Blackboard -- THE shared state surface |
| `blackboard/typed_slot.rs` | TypedSlot -- zero-serde in-process |
| `blackboard/bind_bridge.rs` | SubstrateView trait + BindBridge |
| `blackboard/a2a.rs` | A2ARegistry -- agent discovery |
| `drivers/nars.rs` | NARS evidence-based inference |
| `drivers/spo.rs` | SPO conversation graph |
| `drivers/markov_barrier.rs` | Blood-brain barrier (XOR budget) |
| `agents/`, `meta_agents/` | Agent lifecycle, MetaOrchestrator |
| `chat/handler.rs` | Awareness session pipeline |
| `chat/awareness_session.rs` | xAI REST + caching |
| `llms/providers/anthropic/` | Claude Opus 4.5/4.6 |
| `persona/jit_link.rs` | AgentCard -> ThinkingStyle -> JIT template |
| `persona/thinking_style.rs` | 36 styles in 6 clusters, 23D cognitive space |
| `contract/` | Unified execution contract |
| `mcp/` | Model Context Protocol client |

### 3.2 SubstrateView Trait (THE Contract)

```rust
pub trait SubstrateView: Send + Sync {
    fn read_fingerprint(&self, addr: u16) -> Option<[u64; 256]>;
    fn read_label(&self, addr: u16) -> Option<String>;
    fn read_truth(&self, addr: u16) -> Option<(f32, f32)>;
    fn write_truth(&mut self, addr: u16, frequency: f32, confidence: f32);
    fn hamming_search(&self, query: &[u64; 256], prefix_range: (u8, u8),
                      top_k: usize, threshold: f32) -> Vec<SubstrateMatch>;
    fn write_fingerprint(&mut self, addr: u16, fingerprint: [u64; 256]) -> bool;
    fn xor_delta(&mut self, addr: u16, delta: [u64; 256]);
    fn noise_floor(&self, prefix_range: (u8, u8)) -> f32;
}
```

ladybug-rs implements this for BindSpace. crewai-rust NEVER imports BindSpace directly.

### 3.3 Blackboard

- **JSON slots**: cross-process serialized (MCP/REST boundary)
- **TypedSlots**: in-process zero-serde (`Box<dyn Any>`, moved not copied)
- Canonical keys: `awareness:frame`, `awareness:nars`, `awareness:nars_deltas`, `awareness:spo_triples`
- Phase discipline: only ONE subsystem writes at a time

### 3.4 NARS Driver

Pure-function inference on Blackboard types:
- `nars_analyze(&frame, &axes) -> NarsSemanticState`
- `nars_to_weight_deltas(&state) -> [f32; 32]`
- `build_nars_context(&state) -> Option<String>`

### 3.5 JIT Link Pipeline

```
AgentCard (YAML) -> JitProfile::from_module()
    -> ThinkingStyle (36 styles, 6 clusters, 23D cognitive space)
    -> tau addresses -> JitScanParams { threshold, top_k, prefetch_ahead, filter_mask }
    -> n8n-rs CompiledStyleRegistry -> jitson Cranelift compile
    -> native ScanKernels (compiled at startup, cached)
```

### 3.6 Three-Tier Awareness

1. **Tier 1 (Core)**: BindSpace <-> SubstrateView <-> Blackboard (zero-serde)
2. **Tier 2 (Blood-Brain Barrier)**: MarkovBarrier XOR budget gates state transitions
3. **Tier 3 (External)**: xAI/Grok/Anthropic APIs via n8n-rs workflows

---

## 4. NEO4J-RS -- Property Graph Database

Clean-room Rust reimplementation. ~10K lines. Zero tech debt by design.

### 4.1 Pipeline

```
Parser: &str -> Result<Statement>      (pure function, zero IO)
Planner: AST -> LogicalPlan            (backend-agnostic)
Optimizer: LogicalPlan -> LogicalPlan   (rule-based)
Executor: LogicalPlan + StorageBackend -> QueryResult
```

### 4.2 StorageBackend

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    type Tx: Transaction;
    async fn begin_tx(&self, mode: TxMode) -> Result<Self::Tx>;
    async fn commit_tx(&self, tx: Self::Tx) -> Result<()>;
    // Node CRUD, Relationship CRUD, Index, Expand...
}
```

Implementations: `MemoryBackend` (test), `BoltBackend` (external Neo4j), `LadybugBackend` (production).

### 4.3 LadybugBackend Mapping

| Neo4j | BindSpace |
|---|---|
| Node | BindNode at Addr (0x80-0xFF:XX) |
| Node labels | BindNode.label |
| Node properties | BindNode.payload (JSON) |
| Relationship | BindEdge (from -> verb -> to) |
| NodeId | Addr packed as u64 |

---

## 5. N8N-RS -- Workflow Orchestration

Rust transcode of n8n. 8 crates in `n8n-rust/` workspace.

| Crate | Purpose |
|---|---|
| `n8n-core` | Execution engine: stack-based, resumable, wait nodes, retry, jitson hooks |
| `n8n-workflow` | Workflow types: Workflow, Node, Connection, Execution |
| `n8n-grpc` | Multi-transport: REST, gRPC, Arrow Flight, STDIO |
| `n8n-arrow` | Arrow 57, IPC, Flight, DataFusion 51 |
| `n8n-hamming` | 10kbit Hamming vectors (1,250 bytes per vector) |
| `n8n-db` | PostgreSQL persistence |
| `n8n-contract` | Unified execution contract: crew/ladybug routing, MCP, free will pipeline |
| `n8n-server` | Multi-transport server binary |

### 5.1 n8n-contract: The Bridge

Routes between subsystems via prefix addressing:
- `crew.*` steps -> crewai-rust (prefix 0x0C)
- `lb.*` steps -> ladybug-rs (prefix 0x05)
- `n8n.*` steps -> n8n-rs (prefix 0x0F)
- CogPacket binary wire protocol for internal routing

Modules: `crew_router`, `ladybug_router`, `wire_bridge`, `free_will`, `mcp_inbound`, `thinking_mode`, `chat_session`, `interface_gateway`, `impact_gate`, `semantic_model`.

### 5.2 Arrow Zero-Copy Chain

```
n8n-rs workflow -> Arrow RecordBatch (n8n-arrow, Arrow 57)
    -> Arrow Flight (port 50052, zero-copy IPC)
    -> ladybug-rs BindSpace (ArrowZeroCopy) -> rustynum SIMD
```

---

## 6. AIWAR-NEO4J-HARVEST

221 nodes (5 types), 356 edges (6 types), 12-axis ontology.

CLI: `cypher` (generate .cypher), `neo4j` (direct ingest), `analyze` (stats), `chess-openings` / `chess-evals` / `chess-bridge` (chess knowledge), `live-games` (stonksfish-ada).

---

## Cross-Repo Data Flow (End-to-End)

1. User message arrives at crewai-rust chat handler
2. grok-3-fast felt-parse extracts intent
3. BindSpace -> AwarenessFrame -> Blackboard TypedSlot (hydrate)
4. NARS inference: AwarenessFrame -> NarsSemanticState (pure function)
5. SPO extraction: Conversation -> SpoTriple graph
6. CollapseGate: Evaluate dispersion -> FLOW/HOLD/BLOCK
7. Prompt enrichment: Build qualia-enriched system prompt
8. LLM call: grok-3 deep response (via n8n-rs workflow)
9. Write-back: New TypedSlots -> BindSpace XOR delta (1-2 words of 256 changed)
10. Persistence: Aged nodes flush to Lance cold tier

---

## Aligned Dependencies

| Dependency | Version | Used By |
|---|---|---|
| Arrow | 57 | ladybug-rs, n8n-rs, rustynum-arrow |
| DataFusion | 51 | ladybug-rs, n8n-rs, rustynum-arrow |
| Lance | 2.0-2.1 | n8n-rs, ladybug-rs (vendor) |
| tonic | 0.14 | ladybug-rs, n8n-rs |
| Rust | 1.93 stable | all repos |
| serde | 1 | all repos |
| PyO3 | (bindings) | rustynum |

---

## Build & CI

- Stable Rust 1.93, edition 2021 (core), 2024 (leaf)
- GitHub Actions: lint (fmt+clippy) + test on ubuntu-latest + macos-latest
- `cargo test --workspace --exclude qualia-xor`
- ~225+ tests (rustynum-core ~131, rustynum-arrow ~44, rustyblas ~50)
- Miri: `cargo +nightly miri test -p rustynum-core` (local only)
- Docker (ladybug-rs): `ARG FEATURES="simd,parallel,flight"`

---

## Open Wiring Work

### P0 (Critical)
- [ ] Delete `ladybug-rs/src/core/simd.rs` (348 lines duplicating rustynum's SIMD dispatch)
- [ ] Wire CogRecordView into BindSpace hydration path (eliminate copies in `hydrate_nodes()`)
- [ ] ladybug-rs SubstrateView impl (connect crewai-rust agents without HTTP)
- [ ] Wire JitProfile into ModuleRuntime activation (JitScanParams -> n8n-rs compile)

### P1 (Important)
- [ ] Wire rustyblas GEMM for batch similarity (138 GFLOPS, post-cascade)
- [ ] Wire rustymkl VML for NARS truth scoring (sigmoid, exp, log)
- [ ] In-process delegation in n8n-rs (replace HTTP proxy with vendor-linked calls)
- [ ] Wire BindBridge into server startup for zero-serde awareness hydration
- [ ] BERT embedding model for blood-brain barrier inbound translation

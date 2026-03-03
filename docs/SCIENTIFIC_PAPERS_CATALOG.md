# Scientific Papers & Theoretical Foundations

Papers referenced across the Ada Rust ecosystem, organized by domain.
Generated: 2026-03-03.

---

## 1. Hyperdimensional Computing / Vector Symbolic Architectures

These papers underpin the Fingerprint<256> type, XOR binding, bundling, and the entire HDC/VSA layer.

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Computing with High-Dimensional Vectors** | Kanerva | 2009 | rustynum-core, rustynum-holo | Foundational VSA: binary vectors, XOR bind, majority bundle |
| **Holographic Reduced Representations** | Plate | 1995 | rustynum-holo | Circular convolution binding -> carrier model |
| **What We Mean When We Say "What's the Dollar of Mexico?"** | Kanerva | 2010 | rustynum-core | Analogy via XOR binding algebra |
| **A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing** | Imani et al. | 2019 | rustynum-bnn | Binary HDC classification |
| **Vector Symbolic Architectures as a Computing Framework for Emerging Hardware** | Schlegel et al. | 2022 | rustynum-core | VSA hardware acceleration justification |
| **Language Geometry Using Random Indexing** | Sahlgren et al. | 2008 | qualia_xor | Random indexing for semantic vectors |

**Key insight implemented**: XOR is self-inverse and associative, making it the universal algebra. `A xor B` binds; `(A xor B) xor B` recovers `A`. DeltaLayer exploits this: `old xor new = delta`, `ground xor delta = view`.

---

## 2. Binary Neural Networks

The rustynum-bnn crate implements these architectures.

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **XNOR-Net: ImageNet Classification Using Binary Neural Networks** | Rastegari et al. | 2016 | rustynum-bnn | Binary weight/activation, XNOR+popcount |
| **Bi-Real Net: Enhancing the Performance of 1-bit CNNs** | Liu et al. | 2018 | rustynum-bnn | Real-valued shortcuts in binary networks |
| **RIF-Net: Rethinking Inference-Free Binary Neural Networks** | Zhang et al. | 2025 | rustynum-bnn | `BPReLU`, `BinaryBatchNorm`, `RifCaBlock`, `RifFlowMetrics` |
| **ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions** | Liu et al. | 2020 | rustynum-bnn | Activation distribution reshaping |

**Key insight implemented**: K0/K1 Belichtungsmesser (exposure meter) cascade -- progressive sampling with K0 (64-bit, ~84% reject), K1 (512-bit, ~97.5%), BF16 (~99.7%), Full (exact). Binary cascade as attention mechanism.

---

## 3. CLAM / CAKES / panCAKES (Clustering)

The rustynum-clam crate is a direct implementation of these algorithms.

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **CLAM: Clustering, Learning, and Approximation with Manifolds** | Ishaq et al. | 2019 | rustynum-clam | Core ClamTree implementation |
| **CAKES: CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search** | Ishaq et al. | 2023 | rustynum-clam | DFS Sieve with triangle-inequality pruning |
| **panCAKES: Compressed Accelerated K-nearest-neighbor Entropy-scaling Search** | Ishaq et al. | 2024 | rustynum-clam | Hierarchical XOR-diff compression (5-70x) |
| **CHAODA: CLAM-based Hierarchical Anomaly and Outlier Detection Algorithms** | Ishaq et al. | 2022 | rustynum-clam | Anomaly detection via cluster graph topology |
| **Local Fractal Dimension (LFD) estimation** | (various) | -- | rustynum-clam | Adaptive depth control in ClamTree |

**Key insight implemented**: Triangle inequality eliminates candidates without computing distance. Combined with K0/K1/K2 cascade: triangle prunes geometry, K0 prunes cheaply, survivors get full evaluation.

---

## 4. NARS (Non-Axiomatic Reasoning System)

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Non-Axiomatic Reasoning System** | Wang | 1995-2013 | ladybug-rs, crewai-rust | Truth values <frequency, confidence>, evidence accumulation |
| **NAL: Non-Axiomatic Logic** | Wang | 2006 | crewai-rust/drivers/nars.rs | Revision, deduction, abduction, induction rules |
| **OpenNARS** | Wang et al. | 2016 | crewai-rust | Reference implementation for evidence-based inference |

**Key insight implemented**: NARS truth values replace binary true/false with `<f, c>` where frequency = positive evidence / total evidence, confidence = total evidence / (total + k). CollapseGate uses standard deviation of NARS confidence as its trigger.

---

## 5. BLAS / Linear Algebra / HPC

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Anatomy of High-Performance Matrix Multiplication** | Goto & van de Geijn | 2008 | rustyblas | Goto microkernel: MR=6 x NR=16, cache blocking |
| **BLIS: A Framework for Rapidly Instantiating BLAS Functionality** | Van Zee & van de Geijn | 2015 | rustyblas | Packing strategy, micro-tile design |
| **libxsmm: A High Performance Library for Small Matrix Multiplications** | Heinecke et al. | 2016 | rustynum-core | Fixed-size kernel inspiration for K0/K1/K2 |
| **Mixed-Precision Training** | Micikevicius et al. | 2018 | rustyblas | BF16 GEMM with FP32 accumulation |
| **libCEED: Efficient Extensible Discretization** | Brown et al. | 2021 | rustynum-core | TailBackend trait pattern: orchestration dispatches to backends |

**Key insight implemented**: 138 GFLOPS GEMM on commodity hardware via Goto's blocking strategy adapted to AVX-512 register file. Cache blocking: KC=256 (L1), MC=128 (L2), NC=1024 (L3).

---

## 6. Arrow / Columnar Formats

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Apache Arrow: A Cross-Language Development Platform for In-Memory Analytics** | Apache Foundation | 2016+ | rustynum-arrow, ladybug-rs, n8n-rs | Zero-copy columnar format, 64-byte aligned |
| **Lance: A Columnar Data Format for Computer Vision** | LanceDB | 2023+ | ladybug-rs | Cold-tier persistence, mmap'd access |
| **Apache DataFusion: A Fast, Embeddable, Modular Query Engine** | Lamb et al. | 2024 | rustynum-arrow, ladybug-rs | Cascade scan, CogRecordView integration |

**Key insight implemented**: The zero-copy chain: Lance (mmap) -> Arrow Buffer (64-byte aligned) -> BindSpace (O(1)) -> rustynum SIMD -> Result. No allocation in the hot path.

---

## 7. Graph Databases / Cypher

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Cypher: An Evolving Query Language for Property Graphs** | Francis et al. | 2018 | neo4j-rs | Parser, AST, logical plan |
| **openCypher** | Neo4j Inc. | 2017+ | neo4j-rs | TCK compliance target |
| **GQL (ISO/IEC 39075)** | ISO | 2024 | neo4j-rs | Future query language target |
| **Bolt Protocol Specification** | Neo4j Inc. | -- | neo4j-rs | BoltBackend implementation |

---

## 8. Gabor / Signal Processing

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Gabor wavelets** | Gabor (Dennis) | 1946 | rustynum-holo | Spatial frequency analysis for container lifecycle |
| **Gabor features for texture analysis** | Jain & Farrokhnia | 1991 | rustynum-holo | `gabor_read()` / `gabor_write()` |

---

## 9. Causal Inference

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **The Book of Why** | Pearl | 2018 | ladybug-rs (causal) | Counterfactual reasoning framework |
| **Causal Inference in Statistics: A Primer** | Pearl, Glymour, Jewell | 2016 | ladybug-rs | Structural causal models |
| **Free Causal Discovery: A New Paradigm** | (Ada ecosystem paper) | 2026 | rustynum, ladybug-rs | Factorial similarity decomposition via SIMD |

---

## 10. Cognitive Architecture

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **ACT-R** | Anderson | 1993+ | ladybug-rs (0x04) | Activation-based memory retrieval |
| **Global Workspace Theory** | Baars | 1988 | crewai-rust (Blackboard) | Blackboard as global workspace |
| **The Society of Mind** | Minsky | 1986 | crewai-rust (agents) | Multi-agent architecture |
| **Integrated Information Theory (IIT)** | Tononi | 2004+ | ladybug-rs (MUL) | Meta-Uncertainty Layer, phi-based integration |

---

## 11. Fiber Bundles / Differential Geometry

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **Gauge Equivariant Convolutional Networks** | Cohen et al. | 2019 | neo4j-rs (CausalPath) | Fiber-bundle transport for causal paths |
| **Geometric Deep Learning** | Bronstein et al. | 2021 | ladybug-rs | Equivariant representations |

---

## 12. Quantization / Compression

| Paper | Authors | Year | Used In | How |
|---|---|---|---|---|
| **BFloat16: The Secret to High Performance on Cloud TPUs** | Google | 2019 | rustynum-core | BF16 structured distance (sign/exp/man weighting) |
| **8-bit Optimizers via Block-wise Quantization** | Dettmers et al. | 2022 | rustyblas | INT8 quantization strategy |
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | Frantar et al. | 2023 | rustyblas | INT4 quantization path |

---

## Papers Produced by the Ecosystem

| Paper | Location | Status |
|---|---|---|
| Free Causal Discovery: A New Paradigm | Google Drive / Notion | Draft 2026 |
| Sigma-Significance Scoring | `rustynum/docs/SIGMA_SIGNIFICANCE_PAPER.md` | Draft |
| Cross-Plane Partial Binding | `rustynum/docs/CROSS_PLANE_PAPER.md` | Draft |
| Causal Trajectory | `rustynum/docs/CAUSAL_TRAJECTORY_PAPER.md` | Draft |

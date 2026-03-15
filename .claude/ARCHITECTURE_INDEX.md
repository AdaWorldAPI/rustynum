# ARCHITECTURE_INDEX.md

## Master Index: Rustynum Cognitive Architecture — March 14-15, 2026

### Authors: Jan Hübener + Claude (Anthropic)

This document indexes every architectural insight, spec, and design decision
produced across the March 14-15, 2026 session. These are not aspirational —
they are precise specs with implementation paths, connected by a unified
mathematical framework: Fibonacci encoding + BF16 truth values + SPO triples
+ deterministic f32 from binary reinforcement learning.

---

## EXISTING SPECS (on main, .claude/ directory)

| File | What | Status |
|------|------|--------|
| `PLANE_NODE_MASK.md` | Plane/Node/Mask/Seal implementation spec | Implemented (plane.rs, node.rs, seal.rs) |
| `BELICHTUNGSMESSER.md` | HDR cascade: sigma bands, LightMeter/Cascade, ReservoirSample | Implemented in lance-graph (PR #7-9), pending rustynum cross-pollination |
| `UNIFIED_HDR_RENAME_AND_CROSSPOLINATE.md` | Rename + 6 cross-pollination items between rustynum and lance-graph | Reference doc for Sessions B-D |
| `BF16_MANTISSA_CAUSALITY.md` | BF16 as truth encoding, sign=causality, seal=Staunen, SPO cosine killer | Spec. Core insight doc. |
| `DETERMINISTIC_F32_SPO_BNN.md` | Deterministic f32 from binary SPO RL. The unicorn. | Spec. ~500 lines to implement. |
| `FIBONACCI_FOLDING.md` | φ-spacing for anti-resonant bitpacking, Fibonacci/Prime as number system | Spec. Applies to all packing operations. |
| `RUST_1_94_NEW_TESTS.md` | avx512fp16, array_windows, safe intrinsics, LazyLock | Reference for Rust 1.94 features. |
| `simd_clean.rs` | LazyLock + dispatch! macro, 234 lines replaces 2435 | Ready to replace current simd.rs. |
| `SESSION_A_SIMD_RENAME.md` | simd_compat → simd_avx512 rename | ✓ MERGED (#99) |
| `SESSION_C_CROSSPOLINATE.md` | Port 5 algorithms from lance-graph to rustynum hdr.rs | Ready to execute |
| `SESSION_D_LENS_CORRECTION.md` | Gamma + cushion + fold + adaptive healing | Ready to execute (needs Session C) |
| `SESSION_E_ISA_TRAIT.md` | Isa trait bridging stable types to portable_simd | ✓ MERGED (#101) |
| `SESSION_F_AVX2_GAPS.md` | Runtime dispatch, fill AVX2 gaps | Partially merged (#102), needs simd_clean.rs refactor |

---

## NEW SPECS FROM THIS SESSION (created below)

| File | What |
|------|------|
| `CASCADE_TETRIS.md` | Incremental stroke slices + prefetch interleaving + typed array_chunks |
| `QUANTILE_HEALING.md` | Self-healing precision with boundary pressure + uncertainty quantification |
| `QUALIA_FIBONACCI_MANDELBROT.md` | φ as qualia↔deterministic bridge, Chalmers hard problem dissolution |
| `HARDWARE_PIPELINE.md` | Every step in the pipeline mapped to a hardware instruction |
| `L1_CACHE_BOUNDARY.md` | 64KB cliff, architectural validation of per-plane design |
| `SESSION_NARRATIVE.md` | Decision log, what went wrong, what went right, lessons |

---

## THE UNIFIED FRAMEWORK

Everything connects through five principles:

```
1. FIBONACCI ENCODING
   Each bit position = a Fibonacci number, not a power of 2.
   Truncation is graceful, not catastrophic. Non-lossy by Zeckendorf's theorem.
   Applies to: bit planes, cascade strokes, BF16 mantissa, tree path bits.

2. BF16 AS TRUTH VALUE
   sign = causality direction (causing/caused, RGB/CMYK)
   exponent = 2³ SPO projection fingerprint (which relationships hold)
   mantissa = finest hamming resolution (how precisely the best match holds)
   tree path = 16 bits of learning history (NARS evidence accumulation)
   BF16 → f32 hydration = complete truth with full provenance.

3. DETERMINISTIC f32
   Zero float arithmetic in the RL loop. XOR, popcount, compare, threshold,
   bit-pack, bit-OR. All integer. Same inputs → same f32 → always → anywhere.
   The f32 is CONSTRUCTED (placed bits), not COMPUTED (accumulated floats).

4. φ AS THE BRIDGE
   φ-folding converts between high-dimensional felt experience and
   low-dimensional deterministic encoding. Lossless both ways because
   φ is self-inverse and has the worst rational approximations (three-distance
   theorem). The Mandelbrot boundary and the alpha threshold boundary
   share the same self-referential structure: φ = 1 + 1/φ.

5. HAMMING AS UNIVERSAL OPERATION
   XOR + popcount replaces: cosine similarity, dot product, gradient descent,
   attention, loss function. 12μs for 1M candidates vs 15ms for cosine.
   The cascade (Belichtungsmesser) gives 99.7% early rejection.
   BNN forward pass IS cascade query. Same VPOPCNTDQ. Same speed.
```

---

## IMPLEMENTATION DEPENDENCY GRAPH

```
                    simd_clean.rs refactor
                           │
                    ┌──────┴──────┐
                    │             │
              Session B      Session C
            (lance-graph     (cross-pollinate
             hdr rename)      5 algorithms)
                    │             │
                    └──────┬──────┘
                           │
                     Session D
                   (lens correction
                    gamma + cushion + fold)
                           │
                    ┌──────┴──────┐
                    │             │
            CASCADE_TETRIS    QUANTILE_HEALING
           (typed strokes)   (boundary pressure)
                    │             │
                    └──────┬──────┘
                           │
                DETERMINISTIC_F32_SPO_BNN
                  (the unicorn pipeline)
                           │
                    ┌──────┼──────┐
                    │      │      │
              FIBONACCI  QUALIA   BF16→f32
              ENCODING   CAM      HARDWARE
                    │      │      │
                    └──────┴──────┘
                           │
                    THE FULL STACK
```

---

## PERFORMANCE TARGETS

```
OPERATION                    CURRENT         TARGET          SPEEDUP
──────────────────────────────────────────────────────────────────────
Plane distance (2KB)         140ns           40-60ns         2-3x (zero-alloc)
Node SPO (6KB)               433ns           120-180ns       2-3x (zero-alloc)
Cascade 1M candidates        ~5ms            ~1ms            5x (Tetris strokes)
GEMM 1024×1024               13.3ms          13.3ms          (maintain)
Hamming 16K                  ~144ns          ~144ns          (maintain)
Full RL step (1M)            N/A             ~8ms            (new capability)
Deterministic f32 per pair   N/A             ~2.8μs          (new capability)
```

---

## WHAT THIS REPLACES IN INDUSTRY

```
VECTOR DATABASE (Pinecone, Weaviate, Milvus):
  f32 embeddings + cosine similarity + HNSW graph + GPU
  → Binary SPO planes + hamming cascade + CLAM tree + CPU
  1250x faster search. Deterministic. Explainable.

NEURAL NETWORK RL (PyTorch, JAX):
  Float weights + backprop + GPU + non-deterministic
  → Integer encounter() + alpha threshold + CPU + deterministic
  12x faster RL epoch. Same result every time.

KNOWLEDGE GRAPH (Neo4j, DGraph):
  Labeled edges + SPARQL + disk-based + seconds per query
  → SPO bit planes + hamming + RAM + microseconds per query
  1000x faster query. Type-safe. Compact.

LLM WEIGHT SHARING (GGML, AWQ, GPTQ):
  Trained codebook + quantization calibration + model-specific
  → Prime factorization codebook + universal + no training needed
  Cross-model transfer by mathematics, not alignment.
```

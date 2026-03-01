# JAX-PI → Ladybug-rs: Physics-Informed Causal Discovery

## What JAX-PI Is

JAX-PI (Physics-Informed Neural Networks in JAX) solves partial differential equations (PDEs) by training neural networks that respect physical laws. The key papers it implements:

1. **"Respecting Causality for Training PINNs"** — temporal causal weighting
2. **"When and Why PINNs Fail to Train: Neural Tangent Kernel Perspective"** — NTK-based loss balancing
3. **"Gradient Alignment in PINNs"** — second-order optimization
4. **"Random Weight Factorization"** — improved training of continuous representations
5. **"PirateNets"** — residual adaptive networks

Forked from: `PredictiveIntelligenceLab/jaxpi` (Stanford)

## The Three Isomorphisms

### Isomorphism 1: PDE Solution Domain ↔ Fingerprint Space

JAX-PI solves equations like Navier-Stokes (fluid flow), Burgers (shock waves), Allen-Cahn (phase separation), Kuramoto-Sivashinsky (chaos) on spatial-temporal domains.

The domain is continuous (x, y, t). The solution u(x,y,t) is a field that must satisfy:
```
∂u/∂t + N(u) = 0     (PDE: physics)
u(x,0) = u₀(x)        (initial condition)
u|∂Ω = g(x,t)          (boundary condition)
```

Our domain is discrete (16,384 bits). The "solution" is the causal graph that must satisfy:
```
NARS revision rule      (dynamics: evidence accumulation)
Codebook initialization (initial condition: σ₃-distinct centroids)
σ-band boundaries       (boundary condition: DISCOVERY/HINT/KNOWN)
```

**The correspondence:**
| PDE | Causal Discovery |
|-----|-----------------|
| Spatial domain Ω | Fingerprint Hamming space |
| Time axis t | Evidence accumulation steps |
| Solution field u(x,t) | NARS truth value field (f,c) over fingerprint pairs |
| PDE operator N(u) | SPO factorization operator |
| Boundary conditions | σ-band thresholds |
| Initial conditions | Codebook centroids |
| Residual r = ∂u/∂t + N(u) | NARS prediction error (expected vs observed truth) |

### Isomorphism 2: Causal Weighting ↔ CollapseGate

The paper "Respecting Causality for Training PINNs" is the direct hit.

The problem: when training PINNs on time-dependent PDEs, the network must learn the solution at earlier times before it can learn later times. But standard training samples ALL time points simultaneously, letting the network "cheat" by fitting later times without getting earlier times right.

The solution: **causal weighting**. Split the time domain into chunks. Weight each chunk by how well the PREVIOUS chunks are solved. If early chunks have high residual, later chunks get near-zero weight.

```python
# JAX-PI: causal weighting
self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T
# This is a lower-triangular matrix: later chunks see earlier chunks' errors
# Weights: w_i = exp(-ε * Σ_{j<i} r_j)  where r_j = residual of chunk j
```

**This IS the CollapseGate:**
```rust
// ladybug-rs: CollapseGate
pub enum GateState {
    Flow,   // Evidence strong enough → propagate (weight ≈ 1.0)
    Hold,   // Evidence insufficient → buffer (weight ≈ 0.0)
    Block,  // Contradiction → reject (weight = 0.0, flag error)
}
```

Both enforce temporal/evidential ordering:
- JAX-PI: "don't trust the solution at t=5 if the solution at t=1 is still wrong"
- Ladybug: "don't trust edge confidence after 100 comparisons if evidence after 10 comparisons was contradictory"

The causal tolerance parameter `causal_tol` maps directly to the σ-band threshold. When residual exceeds tolerance, the weight drops to near-zero — which is HOLD/BLOCK.

### Isomorphism 3: NTK-Based Weight Balancing ↔ Multi-Term Faktorzerlegung Balancing

JAX-PI balances multiple loss terms (initial condition loss, boundary loss, PDE residual loss, data loss) using the Neural Tangent Kernel:

```python
# Grad norm weighting: equalize gradient contributions
grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)
mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)
```

We have 8 terms from the Faktorzerlegung (∅, S, P, O, SP, PO, SO, SPO) that must be balanced. ChatGPT's review flagged: "false discovery across many tested interactions." The solution is EXACTLY NTK-style balancing:

```rust
// Proposed: balance 8 Faktorzerlegung terms
pub fn balance_factorial_terms(terms: &FactorialTerms) -> FactorialWeights {
    let norms: [f32; 8] = terms.grad_norms(); // How fast each term changes
    let mean_norm = norms.iter().sum::<f32>() / 8.0;
    FactorialWeights {
        weights: norms.map(|n| mean_norm / n), // Equalize contributions
    }
}
```

This ensures that a dominant main effect (S) doesn't drown out a subtle interaction (SPO). The NTK perspective tells us WHY: if the gradient of one term is 100× larger than another, the optimizer only learns the dominant term. Balancing equalizes the learning rate across all causal levels.

## What We Can Steal

### 1. Causal Temporal Weighting → NARS Evidential Weighting

JAX-PI's causal weighting (lower-triangular matrix, exponential decay) translates directly:

```rust
pub struct CausalWeighting {
    pub num_chunks: usize,      // Evidence accumulation stages
    pub tolerance: f32,          // σ-band threshold
    // Lower-triangular: stage i weighted by cumulative error of stages < i
    pub weights: Vec<f32>,
}

impl CausalWeighting {
    pub fn update(&mut self, residuals: &[f32]) {
        for i in 0..self.num_chunks {
            let cumulative_error: f32 = residuals[..i].iter().sum();
            self.weights[i] = (-self.tolerance * cumulative_error).exp();
        }
    }
}
```

This gives NARS revision a formal temporal ordering guarantee: you can't accumulate high confidence on late evidence if early evidence is contradictory. This is EXACTLY what ChatGPT demanded ("correlated evidence in NARS accumulation overstates certainty").

### 2. Fourier Feature Embeddings → Frequency-Aware Fingerprints

JAX-PI uses Fourier features to help networks learn high-frequency functions:
```python
class FourierEmbs(nn.Module):
    def __call__(self, x):
        kernel = self.param("kernel", normal(scale), (x.shape[-1], embed_dim // 2))
        return jnp.concatenate([jnp.cos(x @ kernel), jnp.sin(x @ kernel)], axis=-1)
```

For fingerprints: different nibble positions carry information at different "frequencies" (some change slowly across the corpus = low frequency, some change rapidly = high frequency). A Fourier feature embedding of nibble positions would let the system explicitly model multi-scale structure.

This connects to the 34 tactics document (#13: Convergent/Divergent Oscillation) — oscillation between detail levels IS frequency analysis.

### 3. Gradient Accumulation → Micro-Batch NARS Revision

JAX-PI supports gradient accumulation for memory-constrained training:
```python
if config.grad_accum_steps > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)
```

For streaming causal discovery: instead of revising NARS truth values on every single comparison (noisy), accumulate K comparisons and revise once with the aggregate evidence. This is a direct solution to the false discovery problem: micro-batching reduces the effective number of statistical tests.

### 4. Weight Factorization → Fingerprint Factorization

JAX-PI's "Random Weight Factorization" decomposes each weight matrix W = diag(g) · V where g is a learned scale vector. This improves conditioning.

For fingerprints: the SPO planes are already a factorization. But within each plane, we could further factorize: S = diag(scale) · S_base, where scale is learned per-bit. This would let the system learn which bits within the Subject plane are more informative (high scale) vs. noise (low scale). It's the attention mechanism from deepmsm, but framed as weight factorization from PINNs.

### 5. Period Embeddings → Cyclic Semantic Structure

JAX-PI enforces periodic boundary conditions with trainable periods:
```python
class PeriodEmbs(nn.Module):
    period: Tuple[float]  # cos(period * x), sin(period * x)
```

For semantics: some relationships are cyclic (cause→effect→cause, seasonal patterns, feedback loops). Period embeddings on the temporal dimension of NARS revision would naturally capture cyclic causal structures without special-casing.

## The Kolmogorov Connection

The most recent JAX-PI paper: "Deep Learning Alternatives of the Kolmogorov Superposition Theorem." Kolmogorov proved that ANY continuous function of N variables can be written as a sum of continuous functions of one variable composed with continuous functions of one variable.

This is directly relevant to the 2³ Faktorzerlegung claim: we claim that the 8-term decomposition captures all causal structure. Kolmogorov says: for continuous functions on the cube [0,1]^3, you need 2*3+1 = 7 univariate functions. We produce 8 terms (1 baseline + 3 main + 3 pairwise + 1 irreducible). The numbers are close but the representations are different (our terms are orthogonal contrasts, Kolmogorov's are compositions).

**Research question:** Is the 2³ Faktorzerlegung a discrete binary analog of Kolmogorov superposition? If so, the irreducible SPO term is the analog of the non-decomposable remainder in Kolmogorov's theorem. This would give the Rung 3 claim a completely independent theoretical foundation from information theory (PID) and statistics (ANOVA).

## Implementation Path

### Phase 1: Steal causal weighting
- Port the lower-triangular exponential decay to NARS revision
- Gate state = threshold on cumulative residual
- This formalizes CollapseGate with PDE theory backing

### Phase 2: Steal NTK balancing
- Apply gradient-norm equalization to the 8 Faktorzerlegung terms
- Prevents dominant main effects from drowning subtle interactions
- Directly addresses false discovery (ChatGPT's critique)

### Phase 3: Steal Fourier features for multi-scale fingerprints
- Learn frequency decomposition of nibble positions
- High-frequency positions = fine detail, low-frequency = structure
- Connects to deepmsm attention masks

### Phase 4: Explore Kolmogorov connection
- Is 2³ = Kolmogorov superposition on the binary cube?
- If yes: independent theoretical foundation for Rung 3
- If no: understand where the analogy breaks

## The Three-Repo Triad

```
deepmsm:  Markov dynamics on molecular configurations (metastable states, transitions)
jaxpi:    PDE dynamics on spatial-temporal domains (causal weighting, NTK balancing)
ladybug:  Causal dynamics on fingerprint space (σ-bands, NARS revision, CollapseGate)
```

All three solve the same meta-problem: discovering structure in high-dimensional dynamical systems. deepmsm gives us diagnostics (VAMPE, CK test, attention). jaxpi gives us training discipline (causal ordering, gradient balancing, multi-scale features). ladybug gives us the substrate (binary fingerprints, O(1) factorization, XOR algebra).

The union is stronger than any one alone.

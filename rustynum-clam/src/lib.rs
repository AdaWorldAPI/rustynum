//! # rustynum-clam
//!
//! CLAM (Clustering, Learning and Approximation of Manifolds) integration
//! for the rustynum fingerprint engine.
//!
//! Implements three capabilities from the URI-ABD research:
//!
//! 1. **CLAM Tree** — divisive hierarchical clustering with LFD estimation
//!    (CHESS, arXiv:1908.08551; CAKES, arXiv:2309.05491)
//!
//! 2. **Triangle-inequality search** — exact k-NN and ρ-NN with d_min/d_max
//!    pruning that replaces heuristic σ-cascade thresholds
//!    (CAKES Depth-First Sieve, Algorithm 6)
//!
//! 3. **panCAKES compression** — hierarchical XOR-diff encoding from cluster
//!    centers, choosing min(unitary, recursive) cost per subtree
//!    (panCAKES, arXiv:2409.12161, Algorithm 2)
//!
//! ## Design
//!
//! - Generic `Distance` trait — plug any metric (Hamming, cosine, edit distance)
//! - `HammingDistance` impl uses rustynum's existing SIMD XOR+POPCNT
//! - Zero external dependencies beyond rustynum-core
//! - Depth-first reordering for O(n) memory (CAKES §2.1.2)
//! - No Arc/Mutex — follows rustynum's split_at_mut blackboard pattern
//!
//! ## References
//!
//! - CHESS:    Ishaq, Student, Daniels. IEEE Big Data 2019. arXiv:1908.08551
//! - CAKES:    Prior, Howard, McLaughlin, Ferguson, Ishaq, Daniels. arXiv:2309.05491
//! - panCAKES: Prior, Howard, Light, Ishaq, Daniels. arXiv:2409.12161
//! - CHAODA:   Ishaq, Howard, Daniels. IEEE Big Data 2021. arXiv:2103.11774

pub mod tree;
pub mod search;
pub mod compress;

pub use tree::{ClamTree, Cluster, Distance, HammingDistance, HammingSIMD, Lfd};
pub use tree::{hamming_batch_simd, hamming_top_k_simd};
pub use search::{RhoNnResult, KnnResult, SearchConfig};
pub use compress::{CompressedTree, CompressionStats};

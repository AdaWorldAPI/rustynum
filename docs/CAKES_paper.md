# CAKES: Let Them Have CAKES — A Cutting-Edge Algorithm for Scalable, Efficient, and Exact Search on Big Data

**arXiv:** 2309.05491v3 (Jan 2025)  
**Authors:** Morgan E. Prior, Thomas J. Howard III, Oliver McLaughlin, Terry Ferguson, Najib Ishaq, Noah M. Daniels  
**Institution:** University of Rhode Island / Tufts University  
**Code:** https://github.com/URI-ABD/clam

---

## Abstract

The Big Data explosion has created a demand for efficient and scalable algorithms for similarity search. While much recent work has focused on approximate k-NN search, exact k-NN search has not kept up. We present CAKES, a set of three novel algorithms for exact k-NN search. CAKES's algorithms are generic over any distance function, and do not scale with the cardinality or embedding dimension of the dataset. Instead, they scale with geometric properties of the dataset — namely, metric entropy and fractal dimension — thus providing immense speed improvements over existing exact k-NN search algorithms when the dataset conforms to the manifold hypothesis. We demonstrate these claims by contrasting performance on a randomly-generated dataset against that on some datasets from the ANN-Benchmarks suite under commonly-used distance functions, a genomic dataset under Levenshtein distance, and a radio-frequency dataset under Dynamic Time Warping distance. CAKES exhibits near-constant running time on data conforming to the manifold hypothesis as cardinality grows, and has perfect recall on data in metric spaces. CAKES also has significantly higher recall than state-of-the-art k-NN search algorithms even when the distance function is not a metric. We conclude that CAKES is a highly efficient and scalable algorithm for exact k-NN search on Big Data. We provide a Rust implementation of CAKES under an MIT license at https://github.com/URI-ABD/clam.

## 1. Introduction

Researchers are collecting data at an unprecedented rate. In many fields, datasets are growing exponentially, and this increase in the rate of data collection outpaces improvements in computing performance as predicted by Moore's Law. This indicates that the performance of computing systems will not keep pace with the growth of data. Often dubbed "the Big Data explosion," this phenomenon has created a need for better algorithms to analyze large datasets. Examples of large datasets include genomic databases, time-series data such as radio frequency signals, and neural network embeddings. Large language models such as GPT and LLAMA-2, and image embedding models are a common source of neural network embeddings. Among biological datasets, SILVA 18S contains ribosomal DNA sequences of approximately 2.25 million genomes; the longest of these individual sequences is 3,712 amino acids, but in a multiple sequence alignment, the length becomes 50,000 letters. Among time-series datasets, the RadioML dataset contains approximately 2.55 million samples of synthetically-generated signals of different modulation modes.

Many researchers are interested in similarity search on these datasets. Similarity search enables a variety of applications, including recommendation and classification systems. As the cardinalities and dimensionalities of datasets have grown, however, efficient and accurate similarity search has become challenging; even state-of-the-art algorithms exhibit a steep tradeoff between recall and throughput.

Given some measure of similarity between data points, there are two common definitions of similarity search: k-nearest neighbor search (k-NN) and ρ-nearest neighbor search (ρ-NN). k-NN search aims to find the k most similar points to a query, while ρ-NN search aims to find all points within a similarity threshold ρ of a query. Previous works have used the term *approximate search* to refer to ρ-NN search, but in this paper, we reserve the term *approximate* for algorithms that do not exhibit perfect recall. In contrast, an *exact* search algorithm exhibits perfect recall.

k-NN search is one of the most ubiquitous classification and recommendation methods in use. Naïve implementations of k-NN search, whose time complexity is linear in the dataset's cardinality, prove prohibitively slow for large datasets. While fast algorithms for k-NN search on large datasets exist, they are often approximate, and while approximate search may be sufficient for some applications, the need for efficient and exact search remains. For example, for a majority voting classifier, approximate k-NN search may agree with exact k-NN search for large values of k, but may be sensitive to local perturbations for smaller values of k. This is especially true when classes are not well-separated. Further, there is evidence that distance functions that do not obey the triangle inequality, such as cosine distance, perform poorly for k-NN search in biomedical settings; this suggests that approximate k-NN search could perform poorly in such contexts.

This paper introduces CAKES (CLAM-Accelerated K-NN Entropy-Scaling Search), a set of three novel algorithms for exact k-NN search. We also compare CAKES against several current algorithms; namely, FAISS, HNSW, and ANNOY, on datasets from the ANN-benchmarks suite. We further benchmark CAKES on a large genomic dataset, the SILVA 18S dataset, using Levenshtein distance on unaligned genomic sequences, and a radio frequency dataset, RadioML, under Dynamic Time Warping distance.

### 1.1 Related Work

Recent k-nearest neighbor search algorithms designed to scale with the exponential growth of data include Hierarchical Navigable Small World (HNSW), Facebook AI Similarity Search (FAISS), Approximate Nearest Neighbors Oh Yeah (ANNOY), and entropy-scaling search. However, some of these algorithms do not provide exact search results.

HNSW is based on skip-list-like navigable small world graphs. Each node in the graph corresponds to a datum, and edges are drawn to nearby data. The graph is navigated by starting at a random node and greedily moving to the closest neighbor of the query. HNSW does not guarantee exact search results; instead, it relies on empirically tuning the index such that the graph has good navigability properties.

FAISS is available in many flavors, but we focus on two: FAISS-Flat and FAISS-IVF. FAISS-Flat performs linear search and always provides exact results; hereafter, we treat it as a baseline for exact search. FAISS-IVF quantizes the dataset using k-means, and uses an inverted index to map a query to the nearest Voronoi cell. The search is then performed within that cell. Because the nearest neighbor might lie across a cell boundary, the algorithm has a tunable parameter `nprobe` that specifies the number of additional adjacent or nearby cells to search. FAISS-IVF does not guarantee exact search results but we allow it to tune the `nprobe` parameter to maximize recall.

ANNOY is based on random projection trees, which partition the data using random hyperplanes. For each query, the tree is traversed, and the closest points in the leaf are examined.

Importantly, as suggested by their name, entropy-scaling search algorithms exhibit asymptotic complexity that scales not with the cardinality of the dataset but with its metric entropy, a measure of the dataset's information content. CHESS provided the first entropy-scaling search algorithm with guaranteed exactness for ρ-NN search and CHAODA used graph induction on CLAM trees for anomaly detection. CAKES is a set of three entropy-scaling algorithms for k-NN search, using a refinement of the clustering algorithm from CHESS. In this paper, we introduce CAKES, implemented in the Rust programming language.

We also provide a theoretical analysis of the time complexity of CAKES's algorithms in Sections 2.2.2 and 2.2.5. These analyses assume, among other things, that a query is drawn from the same distribution as the data; they do not model the behavior of the algorithms on data drawn from a uniform distribution. Given that CAKES's algorithms are intended to be used on datasets with a manifold structure, we believe that this assumption is reasonable. On random, uniformly distributed data, we expect CAKES's algorithms to scale linearly with the cardinality of the dataset; on non-random data conforming to the manifold hypothesis, i.e. data for which the algorithms are designed, we expect them to scale with the metric entropy of the dataset. Thus, for evaluation purposes, it makes more sense to analyze the expected performance of CAKES's algorithms under these assumptions.

## 2. Methods

### 2.1 Local Fractal Dimension (LFD)

We define the local fractal dimension (LFD) at a point q at some length scale in the data as:

**Equation 1:**

```
                    log( |B_X(q, r₁)| / |B_X(q, r₂)| )
    LFD(q, r) = ——————————————————————————————————————————
                              log( r₁ / r₂ )
```

where `B(q, r)` is the set of points contained in the metric ball of radius r centered at a point q in the dataset X.

We use a simplified version of Equation 1 by using a length scale where r₁ = 2·r₂:

**Equation 2:**

```
    LFD(q, r) = log₂( |B(q, r)| / |B(q, r/2)| )
```

Intuitively, LFD measures the rate of change in the number of points in a ball of radius r around a point q as r increases. When the vast majority of points in the dataset have low (≪ D) LFD, we can simply say that the dataset has low LFD. We stress that this concept differs from the embedding dimension of a dataset. To illustrate the difference, consider the SILVA 18S rRNA dataset that contains genomic sequences with unaligned lengths of up to 3,712 base pairs and aligned length of 50,000 base pairs. Hence, the embedding dimension of this dataset is at least 3,712 and at most 50,000; as used in this paper it is 3,712. However, physical constraints (namely, biological evolution and biochemistry) constrain the data to a lower-dimensional manifold within this space. LFD is an approximation of the dimensionality of that lower-dimensional manifold in the "vicinity" of a given point. Section 4.1 discusses this concept on a variety of datasets, showing how real datasets uphold the manifold hypothesis. For real-world datasets, we expect the LFD to be locally uniform, i.e., when r is small, but potentially highly variable at larger length scales, i.e., when r is large.

### 2.1.1 Clustering

We define a cluster as a set of points with a center and a radius. The center is the geometric median of the points in the cluster, i.e., it is the point that minimizes the sum of distances to all other points in the cluster. In cases where the cardinality of the cluster is large, we take a random subsample of √|C| points and compute the geometric median of that subsample. The center, therefore, is one of the points in the cluster and is used as a representative of the cluster. The radius is the maximum distance from the center to any point in the cluster. Each non-leaf cluster has two child clusters in much the same way that a node in a binary tree has two child nodes. Note that clusters can have overlapping volumes and, in such cases, points in the overlapping volume are assigned to exactly one of the overlapping clusters. As a consequence, a cluster can be a proper subset of the metric ball at the same center and radius, i.e., C(c,r) ⊂ B(c,r). We denote the cluster tree by T and the root by R.

Hereafter, when we refer to the LFD of a cluster, it is estimated at the length scale of the cluster radius and half that radius, i.e., using Equation 2. We also only use points that are in C(c,r) instead of all those in B(c,r).

The metric entropy N_r̂(X) for some radius r is the minimum number of clusters of a uniform radius r needed to cover the data for a flat clustering with clusters of radius r. In this paper, where the clustering is hierarchical rather than flat, we define the metric entropy N_r̂(X) as the number of leaf clusters in the tree where r̂ is the mean radius of all leaf clusters.

### 2.1.2 Building the Tree

We start by performing a divisive hierarchical clustering on the dataset using CLAM to obtain a cluster tree T. The procedure is almost identical to that outlined in CHESS, but with better selection of poles for partitioning.

**Algorithm 1: Partition(C, criteria)**
```
Require: f: X×X → R⁺ ∪ {0}, a distance function
Require: C, a cluster
Require: criteria, user-specified continuation criteria

seeds ← random sample of √|C| points from C
c     ← geometric median of seeds
l     ← arg max f(c, x) ∀ x ∈ C
r     ← arg max f(l, x) ∀ x ∈ C
L     ← {x | x ∈ C ∧ f(l, x) ≤ f(r, x)}
R     ← {x | x ∈ C ∧ f(r, x) < f(l, x)}
if |L| > 1 and L satisfies criteria then
    Partition(L, criteria)
end if
if |R| > 1 and R satisfies criteria then
    Partition(R, criteria)
end if
```

Given a cluster C with |C| points, we define its two children by the following process: We take a random subsample S of √|C| of C's points, and compute pairwise distances between all points in S. Using these distances, we compute the geometric median of S; in other words, we find the point that minimizes the sum of distances to all other points in S. We define the center of C to be this geometric median. The radius of C is the maximum distance from the center to any other point in C. The point that is responsible for that radius (i.e., the furthest point from the center) is designated the left pole and the point that is furthest from the left pole is designated the right pole. We then partition the cluster into a left child and a right child, where the left child contains all points in the cluster that are closer to the left pole than to the right pole, and the right child contains all points in the cluster that are closer to the right pole than to the left pole.

### 2.1.3 Depth-First Reordering

In CHESS, each cluster stored a list of indices into the dataset. This list was used to retrieve the clusters' points during search. Although this approach allowed us to retrieve the points in constant time, its memory cost was prohibitively high. With a dataset of cardinality n and each cluster storing a list of indices for its points, we stored a total of n indices at each depth in the tree T. Assuming T is balanced, and thus O(log n) depth, this approach had a memory overhead of O(n log n). In this work, we introduce a new approach wherein, after building T, we reorder the dataset so that points are stored in a depth-first order. Then, within each cluster, we need only store its cardinality and an offset to access its points from the dataset. The root cluster R has an offset of zero and a cardinality equal to the entire dataset. Each non-leaf cluster's children have offsets that are contiguous with the parent's offset, and cardinalities that sum to the parent's cardinality. This approach has a memory overhead of O(1) per cluster, and so the total memory overhead is O(n) for the tree with n leaf clusters.

### 2.2 Search

> **Figure 1 Description:** [CAKES Query Geometry] Shows a cluster C (circle of points with center c) and a query point q outside the cluster. Three distances are illustrated: δ = f(q,c) (blue, distance from query to cluster center), δ⁺ = δ + radius (red, maximum possible distance from q to any point in C), and δ⁻ = δ - radius (green, minimum possible distance from q to any point in C). These bounds are used to prune the search space during tree traversal.

**Algorithm 2: tree-search(C, q, r)**
```
Require: f, a distance function
Require: C, a cluster
Require: q, a query
Require: r, a search radius

if δ⁺_C ≤ r then
    return {C}              // entire cluster is within radius
else
    [L, R] ← children of C
    return tree-search(L, q, r) ∪ tree-search(R, q, r)
end if
```

**Algorithm 3: leaf-search(Q, q, r)**
```
Require: f, a distance function
Require: Q, a set of clusters
Require: q, a query
Require: r, a search radius

H ← ∅
for C ∈ Q do
    if δ⁺_C ≤ r then
        H ← H ∪ C           // add all points
    else
        for p ∈ C do
            if f(p, q) ≤ r then
                H ← H ∪ {p}
            end if
        end for
    end if
end for
return H
```

Given a dataset X, a distance function f and the root cluster R of the tree T (constructed by Algorithm 1), we can now perform ρ-NN search using tree-search (Algorithm 2) followed by leaf-search (Algorithm 3).

### 2.2.1 Three k-NN Algorithms

We present three novel algorithms for exact k-NN search: Repeated ρ-NN, Breadth-First Sieve, and Depth-First Sieve. In these algorithms, we use H (for "hits") to refer to the data structure that stores the closest points to the query found so far. We say that a point "makes the cut" if it has a chance of being one of the k nearest neighbors.

**Algorithm 4: Repeated ρ-NN(R, q, k)**
```
Require: R, the root cluster
Require: q, a query
Require: k, the number of neighbors to find

r ← radius of R
H ← tree-search(R, q, r) then leaf-search(results, q, r)
while |H| < k do
    r ← 2 · r
    H ← tree-search(R, q, r) then leaf-search(results, q, r)
end while
sort H by distance to q
return first k elements of H
```

**Algorithm 5: Breadth-First Sieve(R, q, k)**
```
Require: R, the root cluster
Require: q, a query
Require: k, the number of neighbors to find

c ← center of R
Q ← { (R, δ⁺_R, |R|-1), (c, δ_R, 1) }
while Σ m ≠ k do                    // m = multiplicity
    τ ← QuickSelect(Q, k)
    Q' ← ∅
    for (C, _, _) ∈ Q do
        if δ⁻_C ≤ τ then
            if C is a point then
                Q' ← Q' ∪ {(C, δ_C, 1)}
            else if C is a leaf then
                Q' ← Q' ∪ {(p, δ_p, 1) for p ∈ C}
            else
                [L, R] ← children of C
                l, r ← centers of L, R
                Q' ← Q' ∪ {(L, δ⁺_L, |L|-1), (l, δ_L, 1)}
                Q' ← Q' ∪ {(R, δ⁺_R, |R|-1), (r, δ_R, 1)}
            end if
        end if
    end for
    Q ← Q'
end while
QuickSelect(Q, k)
return first k points in Q
```

**Algorithm 6: Depth-First Sieve(R, q, k)**
```
Require: R, the root cluster
Require: q, a query
Require: k, the number of neighbors to find

Q ← [R], a min-heap by δ⁻
H ← [], a max-heap by δ of size k
while |H| < k or H.peek.δ ≥ Q.peek.δ⁻ do
    while Q.peek is not a leaf do
        C ← Q.pop (the closest cluster)
        [L, R] ← children of C
        Q.push(L)
        Q.push(R)
    end while
    leaf ← Q.pop
    for p ∈ leaf do
        H.push(p)
    end for
end while
return H
```

### 2.2.2 Complexity of Repeated ρ-NN

**Theorem 1.** Let X be a dataset and q a query sampled from the same distribution (i.e., arising from the same generative process) as X. Then time complexity of performing Repeated ρ-NN search on X with query q is:

**Equation 4:**
```
                                                        d
    O( log N_r̂(X)  +  k · (1 + 2·(|Ĉ|/k)^(d-1))    )
       \_________/     \________________________________/
        tree-search              leaf-search
```

where N_r̂(X) is the metric entropy of the dataset, d is the LFD of the dataset, and k is the number of nearest neighbors.

**Proof sketch:** The tree-search and leaf-search stages are considered separately. Tree-search refers to identifying clusters that overlap with the query ball. In CHESS, we showed that the complexity of ρ-NN search is:

**Equation 5:**
```
                                                  d
    O( log N_r̂(X)  +  |B(q,ρ)| · ((ρ+2·r̂)/ρ)  )
       \_________/     \___________________________/
        tree-search            leaf-search
```

### 2.2.3 Complexity of Breadth-First Sieve

**Equation 6:**
```
    O( log N_r̂(X) + L · log(L) )

    where L = O( k · (1 + 2·(|Ĉ|/k)^(d-1))^d )
```

### 2.2.4 Complexity of Depth-First Sieve

**Equation 7:**
```
    O( log N_r̂(X) + L · log(L) )

    where L is as in Equation 6
```

### 2.3 Auto-Tuning

We perform some simple auto-tuning to select the optimal k-NN algorithm to use with a given dataset. We start by taking the center of every cluster at a low depth (e.g., 10) in T as a query. This gives us a small, representative sample of the dataset. Using these clusters' centers as queries, and a user-specified value of k, we record the time taken for k-NN search on the sample using each of the three algorithms. We select the fastest algorithm over all the queries as the optimal algorithm for that dataset and value of k.

### 2.4 Synthetic Data Augmentation

Based on our asymptotic complexity analyses, we expect CAKES to perform well on datasets with low LFD, and for its performance to scale sub-linearly with the cardinality of the dataset. To test this hypothesis, we use some datasets from the ANN-benchmarks suite and synthetically augment them to generate similar datasets with exponentially larger cardinalities. We do the same with a large random dataset of uniformly distributed points in a hypercube. We then compare the performance of CAKES to that of other algorithms on the original datasets and the synthetically augmented datasets.

To elaborate on the augmentation process, we start with an original dataset from the ANN-benchmarks suite. Let X be the dataset, d be its dimensionality, ε be a user-specified noise level, and m be a user-specified integer multiplier. For each datum x ∈ X, we create m-1 new data points within a distance ε of x. We construct a random vector r of d dimensions in the hyper-sphere of radius ε centered at the origin. We then add r to x to get a new point x'. Since ||r|| ≤ ε, we have that ||x - x'|| ≤ ε (i.e., x' is within a distance ε of x). This produces a new dataset X' with |X'| = m·|X|. This augmentation process does not add to the overall topological structure of the dataset, but it does increase its cardinality by a factor of m. This allows us to isolate the effect of cardinality on search performance from that of other factors such as dimensionality, choice of metric, or the topological structure of the dataset.

## 3. Datasets and Benchmarks

### Table 1: Datasets used in benchmarks

| Dataset | Distance Function | Cardinality | Dimensionality |
|---|---|---|---|
| Fashion-MNIST | Euclidean | 60,000 | 784 |
| Glove-25 | Cosine | 1,183,514 | 25 |
| Sift | Euclidean | 1,000,000 | 128 |
| Random | Euclidean | 1,000,000 | 128 |
| SILVA 18S | Levenshtein | 2,224,640 | 3,712 |
| RadioML | Dynamic Time Warping | 97,920 | 1,024 |

All benchmarks were conducted on an Intel Xeon E5-2690 v4 CPU @ 2.60GHz with 512GB RAM. The OS kernel was Manjaro Linux 5.15.164-1-MANJARO. The Rust compiler was Rust 1.83.0, and the Python interpreter version was 3.9.18.

### 3.1 ANN-Benchmark Datasets

We benchmark on a variety of datasets from the ANN-benchmarks suite. For HNSW, ANNOY, and FAISS-IVF, we allow a hyper-parameter search to tune their index for maximum recall. For CAKES, we build the tree and use our auto-tuning approach to select the fastest algorithm for each dataset and cardinality.

### 3.2 Random Datasets and Synthetic Augmentation

In addition to benchmarks on datasets from the ANN-Benchmarks suite, we also benchmarked on synthetic augmentations of these real datasets. In particular, we use a noise tolerance ε = 0.01 and explore the scaling behavior as the cardinality multiplier increases. We also benchmarked on purely randomly-generated datasets of various cardinalities. For this, we used a base cardinality of 1,000,000 and a dimensionality of 128 to match the Sift dataset. This benchmark allows us to isolate the effect of a manifold structure (which we expect to be absent in a purely random dataset) on the performance of the CAKES' algorithms.

### 3.3 Non-ANN-Benchmark Datasets

We benchmark CAKES on the SILVA 18S and RadioML datasets, which use non-standard distance functions (Levenshtein and DTW respectively). These datasets demonstrate CAKES's generality over arbitrary distance functions, something that HNSW, ANNOY, and FAISS are unable to match since they require embedding in a vector space with Euclidean or cosine distance.

## 4. Results

### 4.1 Local Fractal Dimension of Datasets

Since the time complexity of CAKES algorithms scales with the LFD of the dataset, we examine the LFD of each dataset we used for benchmarks.

> **Figure 2 Description:** [LFD vs. Cluster Depth across six datasets] Six subplot panels showing LFD (y-axis) vs. tree depth (x-axis) with percentile bands (5th, 25th, median, 75th, 95th, min, max). Key findings from each panel:
> - **(a) Fashion-MNIST** (dim=784, Euclidean): LFD rises to peak ~6 around depth 15-20, then drops to 1 at max depth ~35. Moderate spread.
> - **(b) Glove-25** (dim=25, Cosine): Consistently low LFD < 3 across all depths (to ~55). Flat, narrow bands — very well-structured data.
> - **(c) Sift** (dim=128, Euclidean): LFD peaks sharply at ~9 around depth 10, then decreases smoothly to leaves at depth ~50. Wider spread than Glove-25.
> - **(d) Random** (dim=128, Euclidean): DRAMATICALLY different. LFD starts at ~20 at depth 0. All percentile lines decrease LINEARLY with depth to ~0 at depth 28. This is the curse of dimensionality in action: LFD ≈ log₂(1,000,000) ≈ 20 for the root.
> - **(e) SILVA 18S** (dim=3712, Levenshtein): Consistently very low LFD < 3, near 1 for depth > 40. Deep tree to depth ~100. Strong manifold structure.
> - **(f) RadioML** (dim=1024, DTW): Three distinct LFD peaks (~12) near depths 8, 25, and 50. Piecewise manifold structure from different modulation modes.

The Fashion-MNIST dataset has an embedding dimension of 784. Until approximately depth 5, its LFD is low (< 4). It then increases, reaching a peak of about 6 near depth 20, before decreasing to 1 at the maximum depth.

The Glove-25 dataset has an embedding dimension of 25 and uses cosine distance (notably, not a metric). Relative to Fashion-MNIST, Glove-25 has low LFD. All percentile lines are flatter and lower, with LFD < 3 for all depths.

The Sift dataset (dim=128, Euclidean) has higher LFD relative to Fashion-MNIST and Glove-25. It increases sharply to a peak of 9 around depth 10, then decreases smoothly until reaching the deepest leaves.

The Random dataset (same cardinality and dimensionality as Sift, uniform distribution) is significantly different. The LFD starts at ~20 at depth 0 and all percentile lines decrease linearly with depth. The LFD of approximately 20 for the root cluster reflects the curse of dimensionality: with high probability, for every point in R, its distance from the center c is greater than r/2. Given our LFD definition, this means LFD(R) ≈ log₂(|X|/1) = log₂(1,000,000) ≈ 20. Theoretically, the LFD of this dataset should be 128 (the embedding dimension); with sample sizes larger than 1,000,000, we would expect the LFD to approach 128.

The SILVA 18S dataset (Levenshtein distance) exhibits consistently low LFD < 3 for all depths, hovering near 1 for clusters at depth 40 and deeper.

The RadioML dataset (DTW distance) shows three distinct peaks around LFD of 12 at depths 8, 25 and 50. This suggests piecewise manifold structure from the different modulation modes present in the dataset.

### 4.2 Scaling Behavior and Recall

> **Figure 3 Description:** [Throughput vs. Cardinality across six datasets, k=10] Six subplot panels showing QPS (queries per second, log scale y-axis) vs. Cardinality (log scale x-axis) for seven algorithms: KnnBreadthFirst, KnnDepthFirst, KnnRepeatedRnn, KnnLinear, HNSW, ANNOY, FAISS-IVF. Recall values annotated above HNSW and ANNOY curves. Key findings:
> - **(a) Fashion-MNIST**: HNSW ~15,000 QPS but recall drops 0.95→0.36. ANNOY ~2,000 QPS, recall 0.95→0.58. CAKES DepthFirst ~500 QPS, constant throughput, recall=1.000 always. FAISS-IVF degrades linearly.
> - **(b) Glove-25**: HNSW ~15,000 QPS, recall 0.80→0.18. ANNOY ~1,000 QPS, recall 0.83→0.63. CAKES DepthFirst ~500 QPS, stable, recall ~1.000*.
> - **(c) Sift**: HNSW recall drops 0.69→0.19. ANNOY recall 0.69→0.64. CAKES stable ~100 QPS with perfect recall.
> - **(d) Random**: ALL algorithms degrade. HNSW/ANNOY recall near 0.01-0.06. CAKES throughput drops linearly — worse than linear scan. This confirms entropy-scaling works only on manifold-structured data.
> - **(e) SILVA 18S** (Levenshtein): Only CAKES algorithms benchmarked (HNSW/ANNOY/FAISS can't use Levenshtein). DepthFirst is fastest at ~50 QPS, scaling gently.
> - **(f) RadioML** (DTW): Only CAKES algorithms. Very slow (< 1 QPS) due to expensive DTW computation. DepthFirst still fastest.

### Table 2: Fashion-MNIST Throughput (QPS) and Recall, k=10

| Mult. | HNSW QPS | HNSW Recall | ANNOY QPS | ANNOY Recall | FAISS-IVF QPS | FAISS-IVF Recall | CAKES QPS | CAKES Recall |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.33×10⁴ | 0.954 | 2.19×10³ | 0.950 | 2.01×10³ | 1.000* | 3.46×10³ | 1.000 |
| 2 | 1.38×10⁴ | 0.803 | 2.12×10³ | 0.927 | 9.39×10² | 1.000* | 3.68×10³ | 1.000 |
| 4 | 1.66×10⁴ | 0.681 | 2.04×10³ | 0.898 | 4.61×10² | 0.997 | 3.44×10³ | 1.000 |
| 8 | 1.68×10⁴ | 0.525 | 1.93×10³ | 0.857 | 2.26×10² | 0.995 | 3.30×10³ | 1.000 |
| 16 | 1.85×10⁴ | 0.493 | 1.91×10³ | 0.862 | 1.36×10² | 0.988 | 3.47×10³ | 1.000 |
| 32 | 1.82×10⁴ | 0.543 | 1.92×10³ | 0.860 | 6.35×10¹ | 0.972 | 3.26×10³ | 1.000 |
| 64 | 1.97×10⁴ | 0.380 | 1.81×10³ | 0.782 | 5.57×10¹ | 0.983 | 3.45×10³ | 1.000 |
| 128 | 2.14×10⁴ | 0.361 | 1.86×10³ | 0.683 | — | — | 3.57×10³ | 1.000 |
| 256 | — | — | 2.09×10³ | 0.593 | — | — | 3.41×10³ | 1.000 |
| 512 | — | — | 1.95×10³ | 0.584 | — | — | 3.48×10³ | 1.000 |

*Note:* 1.000* denotes imperfect recall that rounds to 1.000.

### Table 3: Glove-25 Throughput (QPS) and Recall, k=10

| Mult. | HNSW QPS | HNSW Recall | ANNOY QPS | ANNOY Recall | FAISS-IVF QPS | FAISS-IVF Recall | CAKES QPS | CAKES Recall |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.30×10⁴ | 0.804 | 8.84×10² | 0.833 | 7.63×10² | 0.999 | 5.10×10² | 1.000* |
| 2 | 1.43×10⁴ | 0.613 | 8.67×10² | 0.832 | 3.75×10² | 0.997 | 5.29×10² | 1.000* |
| 4 | 1.46×10⁴ | 0.440 | 7.88×10² | 0.837 | 2.17×10² | 0.996 | 4.76×10² | 0.999 |
| 8 | 1.50×10⁴ | 0.290 | 7.80×10² | 0.833 | 1.15×10² | 0.993 | 4.49×10² | 0.997 |
| 16 | 1.45×10⁴ | 0.209 | 7.63×10² | 0.893 | 6.39×10¹ | 0.980 | 4.70×10² | 1.000* |
| 32 | — | — | 7.49×10² | 0.763 | 2.85×10¹ | 0.952 | 4.56×10² | 1.000* |
| 64 | — | — | 7.84×10² | 0.632 | — | — | 4.46×10² | 1.000* |

### Table 4: Sift Throughput (QPS) and Recall, k=10

| Mult. | HNSW QPS | HNSW Recall | ANNOY QPS | ANNOY Recall | FAISS-IVF QPS | FAISS-IVF Recall | CAKES QPS | CAKES Recall |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.39×10³ | 0.678 | 8.87×10⁰ | 0.997 | 1.34×10² | 1.000 | 1.05×10² | 1.000 |
| 2 | — | — | — | — | — | — | 1.00×10² | 1.000 |
| 4 | — | — | — | — | — | — | 1.03×10² | 1.000 |
| 8 | — | — | — | — | — | — | 9.73×10¹ | 1.000 |
| 16 | — | — | — | — | — | — | 9.66×10¹ | 1.000 |
| 32 | — | — | — | — | — | — | 9.78×10¹ | 1.000 |
| 64 | — | — | — | — | — | — | 9.12×10¹ | 1.000 |
| 128 | — | — | — | — | — | — | 9.33×10¹ | 1.000 |

### Table 5: Random Dataset Throughput (QPS) and Recall, k=10

| Mult. | HNSW QPS | HNSW Recall | ANNOY QPS | ANNOY Recall | FAISS-IVF QPS | FAISS-IVF Recall | CAKES QPS | CAKES Recall |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.36×10¹ | 0.643 | 4.78×10⁰ | 0.993 | 1.31×10² | 1.000 | 2.75×10¹ | 1.000 |
| 2 | — | — | 3.44×10⁰ | — | — | — | — | 1.000 |
| 4 | — | — | 2.09×10⁰ | — | — | — | — | 1.000 |
| 8 | — | — | 1.24×10² | — | — | — | — | 1.000 |

*Note:* On the Random dataset, CAKES throughput degrades linearly with cardinality, confirming that entropy-scaling requires manifold structure.

### 4.3 Key Findings

On the Fashion-MNIST, Glove-25, and Sift datasets, the Depth-First Sieve algorithm is consistently the fastest CAKES algorithm with a throughput that is **near-constant as cardinality grows**. All three CAKES algorithms exhibit perfect recall on Fashion-MNIST and Sift (which use Euclidean distance, a true metric), and near-perfect recall on Glove-25 (which uses cosine distance, not a metric).

In contrast, HNSW and ANNOY are faster than CAKES's algorithms for all cardinalities, but **their recall degrades quickly as cardinality increases**. At 512× augmentation on Fashion-MNIST, ANNOY's recall has dropped to 0.58 while CAKES maintains 1.000.

FAISS-IVF exhibits linearly decreasing throughput as cardinality increases, which is expected given that we tune the hyper-parameters to maximize recall.

On the Random dataset, all algorithms degrade, confirming that CAKES is designed for manifold-structured data.

### 4.4 Balanced vs. Unbalanced Clustering

> **Figure 4 Description:** [Distance Computations across four clustering strategies × three search algorithms on Fashion-MNIST] Compares balanced clustering (via k-means bisection and random bisection) vs. unbalanced clustering (CLAM's approach) on number of distance computations required. Unbalanced clustering consistently requires fewer distance computations across all three CAKES search algorithms, validating the design choice.

## 5. Discussion and Conclusions

We present CAKES, a set of three algorithms for exact k-NN search that scale with the metric entropy and the local fractal dimension of the dataset rather than its cardinality or embedding dimension. On datasets conforming to the manifold hypothesis, CAKES exhibits near-constant throughput as cardinality grows while maintaining perfect recall in metric spaces and near-perfect recall in non-metric spaces (e.g., cosine distance).

CAKES's algorithms are generic over any distance function that satisfies the triangle inequality. For non-metric distance functions, CAKES still achieves significantly higher recall than approximate algorithms. The algorithms are implemented in Rust and are available at https://github.com/URI-ABD/clam under an MIT license.

Key limitations: CAKES is not the fastest algorithm in raw throughput — HNSW and ANNOY are consistently faster. However, CAKES is the only algorithm that maintains perfect or near-perfect recall as cardinality scales, which is critical for applications where exact results matter (e.g., genomic analysis, legal discovery, medical diagnosis).

Future work includes parallelization of the search algorithms, support for streaming/dynamic datasets, and integration with the panCAKES compression framework for compressive search.

---

## Supplement (Summary)

The paper includes a supplementary section with:

- **Improved ρ-NN pruning:** When a cluster overlaps with the query ball, instead of always searching both children, CAKES checks whether the query's projection onto the pole-pole axis is close enough to the bisection midpoint. If not, one child is pruned. This uses the law of cosines to compute the projection distance without explicitly projecting. This check is O(1) per cluster.
- **Detailed proofs** of Theorems 1–3 (complexity bounds for all three k-NN algorithms).
- **Indexing and tuning time benchmarks** (Figure 3 in supplement): CAKES indexing is faster than FAISS-IVF for all cardinalities on all datasets. ANNOY and HNSW have higher indexing times. FAISS-Flat has no indexing cost.
- **k=100 throughput results** (Figure 4 in supplement): Similar patterns to k=10 but with reduced throughput for all algorithms. CAKES algorithms maintain constant throughput on manifold-structured data.
- **Additional LFD analysis** (Figures 5–6): Cluster radius and fractal density distributions vs. depth across all six datasets.

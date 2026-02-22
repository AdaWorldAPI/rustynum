# panCAKES: Generalized Compression and Compressive Search of Large Datasets

**arXiv:** 2409.12161v2 (Dec 2024)  
**Authors:** Morgan E. Prior, Thomas J. Howard III, Emily Light, Najib Ishaq, Noah M. Daniels  
**Institution:** University of Rhode Island / Tufts University  
**Code:** https://github.com/URI-ABD/clam

---

## Abstract

The Big Data explosion has necessitated the development of search algorithms that scale sub-linearly in time and memory. While compression algorithms and search algorithms do exist independently, few algorithms offer both, and those which do are domain-specific. We present panCAKES, a novel approach to compressive search, i.e., a way to perform k-NN and ρ-NN search on compressed data while only decompressing a small, relevant, portion of the data. panCAKES assumes the manifold hypothesis and leverages the low-dimensional structure of the data to compress and search it efficiently. panCAKES is generic over any distance function for which the distance between two points is proportional to the memory cost of storing an encoding of one in terms of the other. This property holds for many widely-used distance functions, e.g. string edit distances (Levenshtein, Needleman-Wunsch, etc.) and set dissimilarity measures (Jaccard, Dice, etc.). We benchmark panCAKES on a variety of datasets, including genomic, proteomic, and set data. We compare compression ratios to gzip, and search performance between the compressed and uncompressed versions of the same dataset. panCAKES achieves compression ratios close to those of gzip, while offering sub-linear time performance for k-NN and ρ-NN search. We conclude that panCAKES is an efficient, general-purpose algorithm for exact compressive search on large datasets that obey the manifold hypothesis. We provide an open-source implementation of panCAKES in the Rust programming language.

## 1. Introduction

Researchers are collecting data at an unprecedented scale. Among biological datasets, the GreenGenes project provides a multiple-sequence alignment of over one million bacterial 16S sequences each 7,682 characters in length, and SILVA 18S contains ribosomal DNA sequences of approximately 2.25 million genomes with an aligned length of 50,000 letters. With these large quantities of high dimensional data, storage and computational costs have replaced sequencing as the primary research bottleneck.

Other fields, especially those which use datasets of set-membership vectors, face different big data challenges. Given that these datasets often have dimensionality on the same order of magnitude as cardinality, the data only sparsely populate the space. Algorithms which do not leverage the sparsity and self-similarity in these datasets are often prohibitively slow.

As the sizes of datasets grow, the ability to compress them has become increasingly valuable. However, researchers who wish to perform analysis on their compressed data face an additional challenge: the computational cost of decompressing the data before analysis. One particular type of analysis used on large datasets in many fields is similarity search. Similarity search enables a variety of applications, including classification systems and genetic sequence analysis.

As described in CAKES, there are two common definitions of similarity search: k-nearest neighbor search (k-NN) and ρ-nearest neighbor search (ρ-NN). Given some measure of similarity between data points (e.g., a distance function), k-NN search aims to find the k most similar points to a query, while ρ-NN search aims to find all points within a similarity threshold ρ of a query.

## 2. Methods

### 2.1 Compression via CLAM Tree

panCAKES leverages the CLAM tree (the same hierarchical binary clustering structure used in CHESS and CAKES) to compress datasets. The key insight is that if nearby points in the data manifold are similar, then one can be encoded cheaply in terms of the other. panCAKES uses two compression strategies:

**Unitary compression:** Each non-center point in a leaf cluster is encoded as the edit sequence (or set difference) from its cluster center. If the cluster is tight (small radius), these encodings are small. The cost is the sum of distances from each non-center point to the cluster center: `C.unitary_cost = Σ f(C.center, x) for x ∈ C`.

**Recursive compression:** For non-leaf clusters, instead of encoding every point in terms of the cluster center, we encode the child cluster centers in terms of the parent center. Specifically, for cluster C with children L and R, child centers ℓ and r, we store the encodings of ℓ and r in terms of C.center, along with the compressed forms of L and R. The cost is: `C.recursive_cost = f(C.center, ℓ) + L.min_cost + f(C.center, r) + R.min_cost`.

**Algorithm 2: Compress(C)**
```
Require: C, a cluster
Require: f, a distance metric

C.unitary_cost ← Σ f(C.center, x) for x ∈ C
C.min_cost ← C.unitary_cost
if C is not a leaf then
    L, R ← C.left_child, C.right_child
    ℓ, r ← L.center, R.center
    Compress(L)
    Compress(R)
    l_cost ← f(C.center, ℓ) + L.min_cost
    r_cost ← f(C.center, r) + R.min_cost
    C.recursive_cost ← l_cost + r_cost
    if C.recursive_cost > C.unitary_cost then
        Delete all descendants of C (turn C into a leaf)
    else
        C.min_cost ← C.recursive_cost
    end if
end if
```

The algorithm traverses from root to leaves computing unitary costs, then on the way back up compares recursive vs. unitary costs. The result is a compressed tree with a mix of unitarily and recursively compressed clusters, where the compression strategy varies by region based on local manifold structure.

> **Figure 1 Description:** [Cluster Tree with Mixed Compression] Shows a binary cluster tree with points {h, x, i, j, k, y, c} at the root. After partitioning: left child (red-shaded, containing h,x,i — unitarily compressed) and right child (clear, containing k,j,y). The right child's leaves j and k are red-shaded (unitarily compressed). Green solid edges from parent→child indicate the child center will be encoded in terms of the parent center (recursive compression). Black dashed edges from a unitarily compressed cluster to its children indicate those children will be deleted during Algorithm 2. Notably, unitarily compressed clusters do not all occur at the same depth — the exact depth at which recursive compression becomes more efficient than unitary compression varies with the local manifold structure.

> **Figure 2 Description:** [Alternate View — Spatial Layout of Compression] Shows the same points in metric space. Center c is in the middle. A solid red edge from x to i indicates unitary compression (i encoded in terms of its cluster center x). Dashed green edges from y to j and y to k indicate recursive compression (j and k encoded in terms of y, the center of their parent cluster).

### 2.2 Search on Compressed Data

panCAKES performs search using the same algorithms introduced in CAKES (ρ-NN search, Repeated ρ-NN, Breadth-First k-NN, Depth-First k-NN), with the key modification that search operates on the compressed tree and only decompresses the portions of the data needed to answer the query. Specifically:

- **ρ-NN Search:** Tree search + leaf search, decompressing leaf clusters only when they overlap with the query ball.
- **Repeated ρ-NN:** Repeatedly performs ρ-NN with increasing radius until k neighbors are found.
- **Breadth-First k-NN Search:** Breadth-first traversal pruning clusters using QuickSelect at each level.
- **Depth-First k-NN Search:** Depth-first traversal using two priority queues for candidates and hits.

### 2.3 Scaling Behavior of Cluster Radii

> **Figure 3 Description:** [Radius Scaling on a 2D Disk] Three diagrams (C₀, C₁, C₂) showing successive partitions of a disk:
> - C₀: Root cluster with center o₀ and radius R₀ (full disk).
> - C₁: After one partition, child cluster has center o₁ and radius R₁, where R₁ < R₀ along one axis but R₀ is preserved along the orthogonal axis. Hence a child cluster CAN have a larger radius than its parent in some dimensions.
> - C₂: After two partitions (consuming both orthogonal axes), R₂ < R₀. After d partitions (d = fractal dimension), radii are guaranteed to decrease by a factor of √2.
>
> This proves that cluster radii decrease after at most d partitions, where d is the local fractal dimension. The multiplicative decrease factor is at least √2 per d levels.

**Equation 1 (LFD Definition):**
```
                    log( |B_X(q, r₁)| / |B_X(q, r₂)| )
    LFD(q, r) = ——————————————————————————————————————————
                              log( r₁ / r₂ )
```

### 2.4 Compression Upper Bound

We provide an upper bound on the cost of compression in terms of LFD and tree depth. If L is the LFD of the dataset, we can think of a "stride" as L partitions (the number needed for radii to decrease by √2). At each stride, the radii decrease, making encodings cheaper. The compression cost per point decreases geometrically with depth.

> **Figure 4 Description:** [Compression Cost Breakdown, LFD=3, stride=1] Shows a cluster tree partitioned into layers. Leaf clusters within the stride (depth 4, red-shaded) are compressed unitarily. Their ancestors (connected by green edges) are compressed recursively. The recursive cost T_R is the sum of encoding child centers in terms of parent centers (green edges). The unitary cost T_U is the sum of encoding all non-center points in terms of their cluster center (red clusters).

## 3. Datasets and Benchmarking

### 3.1 SILVA 18S

The SILVA 18S ribosomal RNA dataset contains 2,224,640 genomes in a multiple sequence alignment (MSA) that is 50,000 characters wide, most of which are gaps or padding. We built the tree using Hamming distance on the pre-aligned sequences. For compression, we store only the indices at which two sequences differ and the characters at those indices. For search, we use Levenshtein distance on the unaligned sequences.

### 3.2 GreenGenes

Two versions: GreenGenes 12.10 (1,075,170 sequences, MSA width 7,682) and GreenGenes 13.8 (1,261,986 unaligned sequences, lengths 1,111–2,368 characters). Clustering and search used Levenshtein distance. Compression used Needleman-Wunsch edit sequences.

### 3.3 PDB-seq

Derived from the Protein Data Bank (PDB). Subset of nucleic-acid sequences with determined structures. Sequences between 30 and 1000 amino acids. Uses Levenshtein distance for clustering and search.

### 3.4 Kosarak

Anonymized click-stream data from a Hungarian news portal. 74,962 sets with members from 27,983 distinct items. Uses Jaccard distance. Compression stores set differences between pairs of sets.

### 3.5 MovieLens 10M

10,000,054 ratings applied to 10,681 movies by 71,567 users. Filtered to 69,363 sets with 65,134 total distinct members. Uses Jaccard distance. Compression and search identical to Kosarak.

## 4. Results

### 4.1 Compression Ratios

### Table I: Compression Ratios

| Dataset | Raw Data (MB) | gzip (MB) | gzip Ratio | panCAKES (Data+Tree) (MB) | panCAKES Ratio |
|---|---|---|---|---|---|
| GreenGenes 13.5 | 1,740 | 256 | 6.80× | 631 + 147 | 2.24× |
| GreenGenes 12.10 | 7,887 | 146 | 54.02× | 289 + 97 | 19.97× |
| SILVA 18S | 107,676 | 4,397 | 24.49× | 1,320 + 219 | 69.96× |
| PDB-seq | 251 | 42 | 5.98× | 326 + 88 | 0.61× |
| Kosarak | 33 | 11 | 3.00× | 10 + 1.5 | 2.87× |
| MovieLens 10M | 63 | 19 | 3.32× | 18 + 1.3 | 3.26× |

**Key findings:**
- panCAKES achieves **3× better compression than gzip on SILVA 18S** (69.96× vs 24.49×), the largest and most self-similar dataset.
- On GreenGenes 12.10, panCAKES achieves 19.97× compression vs gzip's 54.02× — still excellent.
- On PDB-seq, panCAKES actually **expands** the data (0.61×). This is because PDB exhibits selection bias: it is unusual for deposited protein structures to have high sequence similarity, so the self-similarity that panCAKES exploits is absent.
- On set datasets (Kosarak, MovieLens), panCAKES compression is comparable to gzip.

### 4.2 Search Performance

### Table II: Search Time on Raw Data (seconds per query)

| Dataset | ρ-NN | Repeated ρ-NN | BFS k-NN | DFS k-NN |
|---|---|---|---|---|
| GreenGenes 13.5 | 2.53 | 9.64 | 11.71 | 11.96 |
| GreenGenes 12.10 | 3.17 | 21.52 | 24.43 | 18.72 |
| SILVA 18S | 5.87 | 43.48 | 35.59 | 154.66 |
| PDB-seq | 0.67 | 2.07 | 0.91 | 2.63 |
| Kosarak | 6.20×10⁻³ | — | 2.56×10⁻³ | 1.75×10⁻³ |
| MovieLens 10M | 7.38×10⁻³ | — | 2.83×10⁻² | 2.60×10⁻² |

### Table III: Compressive Search Time (seconds per query)

| Dataset | ρ-NN | Repeated ρ-NN | BFS k-NN | DFS k-NN |
|---|---|---|---|---|
| GreenGenes 13.5 | 41.78 | 86.68 | 76.12 | 50.04 |
| GreenGenes 12.10 | 89.57 | 144.59 | 155.31 | 102.20 |
| SILVA 18S | 77.91 | 161.32 | 166.29 | 130.26 |
| PDB-seq | 0.85 | 2.48 | 0.82 | 0.72 |
| Kosarak | 2.10×10⁻² | — | 2.68×10⁻² | 2.84×10⁻² |
| MovieLens 10M | 3.28×10⁻² | — | 5.00×10⁻¹ | 4.94×10⁻¹ |

### Table IV: Slowdown Factor (Compressed / Uncompressed Search Time)

| Dataset | ρ-NN | Repeated ρ-NN | BFS k-NN | DFS k-NN |
|---|---|---|---|---|
| GreenGenes 13.5 | 16.49× | 8.99× | 6.50× | 4.18× |
| GreenGenes 12.10 | 28.26× | 6.72× | 6.36× | 5.46× |
| SILVA 18S | 13.29× | 3.71× | 4.67× | 0.84× |
| PDB-seq | 1.28× | 1.20× | 0.90× | 0.27× |
| Kosarak | 3.39× | — | 10.47× | 16.23× |
| MovieLens 10M | 4.44× | — | 17.67× | 19.00× |

**Key finding:** panCAKES trades search speed for compression. On genomic datasets (GreenGenes, SILVA), ρ-NN search slows by 13–28×, but DFS k-NN shows much smaller slowdowns (4–5×). Remarkably, on SILVA 18S, DFS k-NN is actually **faster** on compressed data (0.84× = 16% speedup), likely because the compressed representation improves cache behavior. On PDB-seq, DFS k-NN compressed is 3.7× faster than uncompressed (0.27×). The primary advantage of panCAKES is that **search happens without decompressing the entire dataset**, which is essential when the dataset is vastly larger than what can fit in system memory.

## 5. Discussion and Future Work

We have presented panCAKES, a novel approach to compression and compressive-search for big data. Our approach allows for efficient similarity search *without decompressing the entire dataset*. We achieve this result by encoding each data point as its set of differences from some representative data point; these differences can then be applied to the representative to reconstruct the original data. This reconstruction need not be applied to the entire dataset (i.e., the root of the CLAM tree), but can instead be applied to any arbitrary subtree corresponding to the result set. The approach is generic over any distance function for which the distance between two points is proportional to the memory cost of storing an encoding of one in terms of the other. In essence, panCAKES can be thought of as a simultaneous database and compression index, allowing for selective decompression based on a query.

Our results illustrate the tradeoff between compression ratios and search speed. Though panCAKES had the greatest compression ratios on the two aligned genomic datasets (SILVA and GreenGenes 12.10), these datasets exhibited relatively slow search times. Notably, panCAKES achieves 3× better compression on SILVA than gzip. While panCAKES compression is comparable to that of gzip on the two set datasets (Kosarak and MovieLens), k-NN search is only mildly slower on the compressed representations of these datasets than on the raw data. We stress that the primary advantage of panCAKES is that search can happen *without decompressing the whole dataset*, which is not possible with general purpose compression algorithms such as gzip. This is essential when the dataset is vastly larger than what can fit in system memory.

We expect that the compression ratio is highly dependent on the amount of self-similarity present in the dataset. This is supported by the fact that SILVA-18S, which is known to have many nearly redundant sequences, exhibits the highest compression ratio of all datasets.

The protein data bank sequences (PDB) form an unusual case, in that PDB exhibits selection bias; it is unusual for protein structures to be deposited when they exhibit high sequence similarity to existing entries. Thus, the advantages of self-similar (and thus, compressible) sequences are elusive here.

Future work includes streaming compression for dynamic datasets, improved parallelization, and integration of compression-aware search into the CAKES auto-tuning framework.

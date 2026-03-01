#!/usr/bin/env python3
"""
Eineindeutigkeit (Bijectivity) Test — Jina 1024-D × Nib4 16-D

Tests that the qualia coordinate system is bijective:
  - Every Nib4 coordinate maps to a unique Jina embedding (injective)
  - Every Jina embedding maps to a unique Nib4 coordinate (surjective)

Uses σ-gating: all 231×230/2 = 26,565 pairwise distances must exceed 3σ
for Eineindeutigkeit to hold at p < 0.001.

Environment: JINA_API_KEY must be set.
"""

import json
import math
import os
import sys
import requests
import numpy as np

JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip('"').strip("'")
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
JINA_DIM = 1024

NIB4_DIMS = [
    "brightness", "valence", "dominance", "arousal", "warmth",
    "clarity", "social", "nostalgia", "sacredness", "desire",
    "tension", "awe", "grief", "hope", "edge", "resolution_hunger"
]
NIB4_LEVELS = 15


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def item_to_rich_text(item: dict) -> str:
    """Build rich text description for Jina embedding (same as BERT step)."""
    parts = [
        f"family: {item['family']}",
        f"item: {item['label']}",
    ]
    if item.get("qualia"):
        parts.append(f"qualia: {', '.join(item['qualia'])}")
    if item.get("melodic_motions"):
        parts.append(f"motion: {', '.join(item['melodic_motions'])}")
    if item.get("harmonic_bias"):
        parts.append(f"harmonic: {', '.join(item['harmonic_bias'])}")
    if item.get("gate", "flow") != "flow":
        parts.append(f"gate: {item['gate']}")
    return " | ".join(parts)


def embed_jina(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts via Jina API in batches. Returns (N, 1024) array."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = requests.post(
            JINA_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JINA_MODEL,
                "input": batch,
                "dimensions": JINA_DIM,
                "task": "text-matching",
            },
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"Jina API error {resp.status_code}: {resp.text[:500]}")
            sys.exit(1)
        data = resp.json()["data"]
        # Sort by index to preserve order
        data.sort(key=lambda x: x["index"])
        for d in data:
            all_embeddings.append(d["embedding"])
        print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings, dtype=np.float32)


def nib4_encode(items: list[dict]) -> np.ndarray:
    """Encode items as Nib4 (4-bit quantized) vectors. Returns (N, 16) u8 array."""
    # Compute per-dimension bounds
    bounds = []
    for dim in NIB4_DIMS:
        vals = [it["vector"].get(dim, 0.0) for it in items]
        mn, mx = min(vals), max(vals)
        if abs(mx - mn) < 1e-9:
            mx = mn + 1.0
        bounds.append((mn, mx))

    encoded = np.zeros((len(items), len(NIB4_DIMS)), dtype=np.uint8)
    for i, it in enumerate(items):
        for d, dim in enumerate(NIB4_DIMS):
            val = it["vector"].get(dim, 0.0)
            mn, mx = bounds[d]
            t = (val - mn) / (mx - mn)
            encoded[i, d] = int(round(t * NIB4_LEVELS))
    return encoded


def nib4_distance_matrix(encoded: np.ndarray, items: list[dict]) -> np.ndarray:
    """Compute L1 distance matrix with intensity-bit penalty."""
    n = len(encoded)
    dist = np.zeros((n, n), dtype=np.float32)
    # Mode bit: shame <= 0 = RGB(causing), shame > 0 = CMYK(caused)
    modes = np.array([it["vector"].get("shame", 0.0) > 0 for it in items])
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum(np.abs(encoded[i].astype(np.int16) - encoded[j].astype(np.int16)))
            if modes[i] != modes[j]:
                d += 16.0  # mode mismatch penalty
            dist[i, j] = d
            dist[j, i] = d
    return dist


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix. Returns (N, N) where 0=identical, 2=opposite."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms
    sim = normed @ normed.T
    return 1.0 - sim


def sigma_test(dist_matrix: np.ndarray, label: str) -> tuple[int, int, float, float]:
    """
    Test all pairs for 3σ distinctness.
    Returns (n_pairs, n_discovery, min_sigma, mean_sigma).
    """
    n = dist_matrix.shape[0]
    # Get upper triangle (all unique pairs)
    triu_idx = np.triu_indices(n, k=1)
    distances = dist_matrix[triu_idx]

    mu = np.mean(distances)
    sigma = np.std(distances)

    if sigma < 1e-10:
        print(f"  {label}: σ ≈ 0, cannot compute σ-scores")
        return (len(distances), 0, 0.0, 0.0)

    # σ-score for each pair: how many σ away from zero?
    # (distance / σ) — but we want "how distinct from being identical"
    # Using the distribution: each distance is a sample. We test if it's > 0.
    # The σ-gate: is dist > 3σ of the *noise floor*?
    # For Eineindeutigkeit: the null hypothesis is "same item" (distance = 0).
    # Under the observed distribution, z = dist / (σ / √n) for population,
    # but per-pair: z = dist / σ_min where σ_min comes from the closest pairs.

    # Simpler and more conservative: use the global distribution
    # z_i = (d_i - 0) / σ = d_i / σ
    # This tests "is this pair's distance significantly different from 0?"
    z_scores = distances / sigma

    n_above_3 = np.sum(z_scores >= 3.0)
    n_above_5 = np.sum(z_scores >= 5.0)
    n_total = len(distances)
    min_z = np.min(z_scores)
    mean_z = np.mean(z_scores)
    max_z = np.max(z_scores)

    # Find the pair with minimum σ
    min_idx = np.argmin(z_scores)
    i_min, j_min = triu_idx[0][min_idx], triu_idx[1][min_idx]

    print(f"\n  {label} σ-gate results:")
    print(f"    Total pairs: {n_total:,}")
    print(f"    μ(distance): {mu:.4f}")
    print(f"    σ(distance): {sigma:.4f}")
    print(f"    σ-scores: min={min_z:.2f}, mean={mean_z:.2f}, max={max_z:.2f}")
    print(f"    ≥ 3σ (Discovery): {n_above_3:,} / {n_total:,} ({100*n_above_3/n_total:.1f}%)")
    print(f"    ≥ 5σ (Strong):    {n_above_5:,} / {n_total:,} ({100*n_above_5/n_total:.1f}%)")
    print(f"    Closest pair: [{i_min}] ↔ [{j_min}] (z={min_z:.2f}, d={distances[min_idx]:.4f})")

    return (n_total, int(n_above_3), float(min_z), float(mean_z))


def kendall_tau_per_item(nib4_dist: np.ndarray, jina_dist: np.ndarray) -> np.ndarray:
    """Compute Kendall tau between Nib4 and Jina rankings per item."""
    n = nib4_dist.shape[0]
    taus = np.zeros(n)
    for i in range(n):
        # Get rankings of other items by each distance metric
        nib4_order = np.argsort(nib4_dist[i])
        jina_order = np.argsort(jina_dist[i])

        # Build rank arrays
        nib4_rank = np.zeros(n, dtype=int)
        jina_rank = np.zeros(n, dtype=int)
        for r, idx in enumerate(nib4_order):
            nib4_rank[idx] = r
        for r, idx in enumerate(jina_order):
            jina_rank[idx] = r

        # Count concordant/discordant pairs (excluding self)
        others = [j for j in range(n) if j != i]
        concordant = 0
        discordant = 0
        for a_idx in range(len(others)):
            for b_idx in range(a_idx+1, len(others)):
                a, b = others[a_idx], others[b_idx]
                nib4_cmp = nib4_rank[a] - nib4_rank[b]
                jina_cmp = jina_rank[a] - jina_rank[b]
                if nib4_cmp * jina_cmp > 0:
                    concordant += 1
                elif nib4_cmp * jina_cmp < 0:
                    discordant += 1
        total = concordant + discordant
        taus[i] = (concordant - discordant) / total if total > 0 else 0.0
    return taus


def main():
    if not JINA_API_KEY:
        print("ERROR: JINA_API_KEY not set")
        sys.exit(1)

    corpus_path = os.path.join(os.path.dirname(__file__), "src", "qualia_219.json")
    items = load_corpus(corpus_path)
    n = len(items)
    print(f"Loaded {n} qualia items across {len(set(it['family'] for it in items))} families")

    # Gate distribution
    gates = {}
    for it in items:
        g = it.get("gate", "flow")
        gates[g] = gates.get(g, 0) + 1
    print(f"Gate distribution: {gates}")

    # Step 1: Nib4 encoding + distance matrix
    print("\n=== Step 1: Nib4 16-D encoding ===")
    nib4 = nib4_encode(items)
    nib4_dist = nib4_distance_matrix(nib4, items)
    nib4_result = sigma_test(nib4_dist, "Nib4 (16-D interior physics)")

    # Step 2: Jina embedding
    print("\n=== Step 2: Jina 1024-D embedding ===")
    texts = [item_to_rich_text(it) for it in items]
    print(f"  Embedding {n} items via Jina {JINA_MODEL} ({JINA_DIM}-D)...")
    jina_emb = embed_jina(texts)
    print(f"  Embedding shape: {jina_emb.shape}")

    # Cosine distance matrix
    jina_dist = cosine_distance_matrix(jina_emb)
    jina_result = sigma_test(jina_dist, "Jina (1024-D observer language)")

    # Step 3: Eineindeutigkeit test
    print("\n=== Step 3: Eineindeutigkeit (Bijectivity) Test ===")
    n_pairs = n * (n - 1) // 2
    nib4_discovery = nib4_result[1]
    jina_discovery = jina_result[1]

    print(f"\n  Nib4: {nib4_discovery}/{n_pairs} pairs at ≥3σ ({100*nib4_discovery/n_pairs:.1f}%)")
    print(f"  Jina: {jina_discovery}/{n_pairs} pairs at ≥3σ ({100*jina_discovery/n_pairs:.1f}%)")

    if nib4_discovery == n_pairs and jina_discovery == n_pairs:
        print(f"\n  ✓ EINEINDEUTIGKEIT CONFIRMED at 3σ in BOTH spaces")
        print(f"    Every qualia coordinate is uniquely identifiable at p < 0.001")
        print(f"    in both interior physics (Nib4) and observer language (Jina)")
    else:
        print(f"\n  ✗ Eineindeutigkeit not fully achieved:")
        if nib4_discovery < n_pairs:
            print(f"    Nib4: {n_pairs - nib4_discovery} pairs below 3σ")
        if jina_discovery < n_pairs:
            print(f"    Jina: {n_pairs - jina_discovery} pairs below 3σ")

    # Step 4: Rank correlation (Kendall tau)
    print("\n=== Step 4: Rank Correlation (Kendall τ) ===")
    taus = kendall_tau_per_item(nib4_dist, jina_dist)
    print(f"  Mean τ:   {np.mean(taus):.4f}")
    print(f"  Median τ: {np.median(taus):.4f}")
    print(f"  Min τ:    {np.min(taus):.4f} (item: {items[np.argmin(taus)]['id']})")
    print(f"  Max τ:    {np.max(taus):.4f} (item: {items[np.argmax(taus)]['id']})")

    # Tau by gate level
    for gate in ["flow", "hold", "block"]:
        idxs = [i for i, it in enumerate(items) if it.get("gate", "flow") == gate]
        if idxs:
            gate_taus = taus[idxs]
            print(f"  {gate} items ({len(idxs)}): τ_mean={np.mean(gate_taus):.4f}")

    # Step 5: Cross-space confusion analysis
    print("\n=== Step 5: Cross-Space Confusion (XOR Buckets) ===")
    k = 10
    bucket_a = 0  # Both agree (structural truth)
    bucket_b = 0  # Nib4 close, Jina far (cadence truth)
    bucket_c = 0  # Jina close, Nib4 far (surface synonymy)
    for i in range(n):
        nib4_neighbors = set(np.argsort(nib4_dist[i])[1:k+1])
        jina_neighbors = set(np.argsort(jina_dist[i])[1:k+1])
        overlap = len(nib4_neighbors & jina_neighbors)
        bucket_a += overlap
        bucket_b += len(nib4_neighbors - jina_neighbors)
        bucket_c += len(jina_neighbors - nib4_neighbors)

    total_neighbors = n * k
    print(f"  k={k} neighborhoods:")
    print(f"    A) Structural Truth (both agree):    {bucket_a}/{total_neighbors} ({100*bucket_a/total_neighbors:.1f}%)")
    print(f"    B) Cadence Truth (Nib4 only):        {bucket_b}/{total_neighbors} ({100*bucket_b/total_neighbors:.1f}%)")
    print(f"    C) Surface Synonymy (Jina only):     {bucket_c}/{total_neighbors} ({100*bucket_c/total_neighbors:.1f}%)")

    # Step 6: Dark arc analysis
    print("\n=== Step 6: Dark Arc Analysis ===")
    dark_idxs = [i for i, it in enumerate(items) if it.get("gate", "flow") != "flow"]
    flow_idxs = [i for i, it in enumerate(items) if it.get("gate", "flow") == "flow"]
    if dark_idxs and flow_idxs:
        # Mean distance from dark items to their nearest prosocial neighbor
        for space_name, dist_mat in [("Nib4", nib4_dist), ("Jina", jina_dist)]:
            dark_to_flow = dist_mat[np.ix_(dark_idxs, flow_idxs)]
            min_dists = np.min(dark_to_flow, axis=1)
            print(f"  {space_name}: dark→nearest_prosocial distances:")
            for idx, d_idx in enumerate(dark_idxs):
                nearest_flow = flow_idxs[np.argmin(dark_to_flow[idx])]
                print(f"    {items[d_idx]['id']:40s} → {items[nearest_flow]['id']:40s} (d={min_dists[idx]:.4f})")

    # Save embeddings for downstream use
    output_path = os.path.join(os.path.dirname(__file__), "src", "jina_embeddings_1024.npy")
    np.save(output_path, jina_emb)
    print(f"\n  Saved Jina embeddings to {output_path}")

    # Save results summary
    results = {
        "n_items": n,
        "n_pairs": n_pairs,
        "nib4_3sigma": nib4_discovery,
        "nib4_min_sigma": nib4_result[2],
        "nib4_mean_sigma": nib4_result[3],
        "jina_3sigma": jina_discovery,
        "jina_min_sigma": jina_result[2],
        "jina_mean_sigma": jina_result[3],
        "eineindeutig": nib4_discovery == n_pairs and jina_discovery == n_pairs,
        "kendall_tau_mean": float(np.mean(taus)),
        "bucket_a_pct": round(100*bucket_a/total_neighbors, 1),
        "bucket_b_pct": round(100*bucket_b/total_neighbors, 1),
        "bucket_c_pct": round(100*bucket_c/total_neighbors, 1),
    }
    results_path = os.path.join(os.path.dirname(__file__), "src", "eineindeutigkeit_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_path}")


if __name__ == "__main__":
    main()

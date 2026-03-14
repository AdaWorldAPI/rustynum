# BF16_MANTISSA_CAUSALITY.md

## BF16 as Distributed Precision: The Tree IS the Mantissa

**Status:** Insight capture. Not a session prompt. Reference when building CLAM leaf compression.

---

## THE CORE INSIGHT

BF16 is f32 with 16 mantissa bits removed. Those bits aren't lost.
They're distributed into the tree structure. The leaf holds WHAT.
The tree path holds WHY. Together they reconstruct full f32.

```
f32:    [8 exponent][23 mantissa]           full value
BF16:   [8 exponent][ 7 mantissa]           coarse value (same range, less precision)
                     └── 16 bits "missing" ──┘

Missing bits = encoded in the tree PATH that leads to this leaf.
Each branch decision = ~1 bit of mantissa resolution.
16 levels of tree depth = 16 bits of distributed mantissa.
```

---

## WHY THIS WORKS

BF16 and f32 share the same 8-bit exponent. Same range: ±3.4×10³⁸.
The only difference is mantissa precision: 7 bits vs 23 bits.

When you INSERT an f32 value into a tree as a BF16 leaf:
- The top 7 mantissa bits go into the leaf (the WHAT)
- The bottom 16 mantissa bits determined which branch the leaf took
  at each level of the tree (the WHY / the causal path)

When you HYDRATE a BF16 leaf back to f32:
- Read the 7 mantissa bits from the leaf
- Walk from root to leaf, each branch decision contributes ~1 bit
- The branch decisions reconstruct the bottom 16 mantissa bits
- Result: full f32 precision, without storing 32 bits per leaf

---

## PRECISION DEPTH = TREE DEPTH

```
Root:         [8 exp][7 mantissa]                         BF16 coarse
  Level 1:    [8 exp][7 mantissa][1 bit from branch]      +1 bit
  Level 2:    [8 exp][7 mantissa][2 bits from path]       +2 bits
  ...
  Level 16:   [8 exp][7 mantissa][16 bits from full path] = f32 exact
```

You don't need all 16 levels for useful precision:
- 4 levels → BF16 + 4 bits = 11-bit mantissa (good enough for search)
- 8 levels → BF16 + 8 bits = 15-bit mantissa (FP16 equivalent)
- 16 levels → BF16 + 16 bits = 23-bit mantissa (full f32)

The cascade can STOP at any level once confidence is sufficient.
Partial hydration = partial precision = enough for rejection.

---

## CONNECTION TO EXISTING ARCHITECTURE

### Plane Alpha Channel

```
BF16 mantissa:  [defined][defined][..unknown..]
Plane alpha:    [1][1][1][0][0][0]...

Same principle:
  Defined bits  = enough evidence to commit
  Unknown bits  = ask the tree / accumulate more encounters
```

### HDR Cascade

```
Cascade Stroke 1:  reads 1/16 of vector  → coarse projection (like BF16 precision)
Cascade Stroke 2:  reads 1/4 of vector   → refined projection (like BF16 + 4 levels)
Cascade Stroke 3:  reads full vector     → exact distance (like full f32)

The cascade IS precision hydration:
  Less data read → less mantissa → coarse decision
  More data read → more mantissa → precise decision
```

### Inverse Causality

Reading the f32 mantissa bits in order tells you the causal chain:

```
Top 7 bits (BF16):     CAUSED — what IS this value
Bottom 16 bits (path):  CAUSALITY — the decisions that produced it

f32 = WHAT + WHY in one 32-bit word.
BF16 = WHAT only. Compact. Storable. Searchable by coarse value.
Full f32 = reconstructible on demand from BF16 + tree path.
```

---

## HARDWARE RELEVANCE

```
f32 → BF16 (INSERT):  VCVTNEPS2BF16 (avx512bf16, stable since Rust 1.89)
                       Hardware rounds to nearest even.
                       32 conversions per instruction.
                       The "lost" 16 bits become tree structure.

BF16 → f32 (HYDRATE): (bits as u32) << 16
                       Lossless reconstruction of the coarse value.
                       Then fill bottom 16 bits from tree path.
                       No hardware instruction needed for the shift.
                       The tree traversal IS the remaining computation.
```

---

## CLAM TREE APPLICATION

```rust
struct ClamLeaf {
    value: BF16,  // 16 bits: coarse value (same exponent range as f32)
    // The remaining 16 bits of f32 precision are implicit
    // in this leaf's POSITION within the tree.
}

/// Compress f32 to BF16 leaf. Precision distributed into tree structure.
fn insert(tree: &mut ClamTree, value: f32) -> LeafId {
    let leaf = BF16::from_f32(value);  // VCVTNEPS2BF16 on capable hardware
    // The branch decisions during insertion encode the bottom 16 bits.
    // They're not stored explicitly — the tree structure IS the storage.
    tree.insert(leaf)
}

/// Reconstruct f32 from BF16 leaf + tree path.
fn hydrate(tree: &ClamTree, leaf_id: LeafId) -> f32 {
    let leaf = tree.get(leaf_id);
    let path = tree.path_to(leaf_id);
    
    let base_bits = (leaf.value.to_bits() as u32) << 16;  // top 16 bits
    let path_bits = path.encode_as_mantissa();              // bottom 16 bits
    f32::from_bits(base_bits | path_bits)
}

/// Partial hydration: reconstruct with N bits of path precision.
/// Useful for cascade search — don't hydrate more than needed.
fn hydrate_partial(tree: &ClamTree, leaf_id: LeafId, depth: usize) -> f32 {
    let leaf = tree.get(leaf_id);
    let path = tree.path_to_depth(leaf_id, depth);
    
    let base_bits = (leaf.value.to_bits() as u32) << 16;
    let path_bits = path.encode_as_mantissa();  // only `depth` bits filled
    f32::from_bits(base_bits | path_bits)
    // Remaining bits are zero — same as BF16 with progressive refinement
}
```

---

## THE PUNCHLINE

The tree doesn't LOSE precision when compressing to BF16.
It DISTRIBUTES the precision across its own structure.
The leaf holds the WHAT. The tree holds the WHY.
Together they reconstruct the full f32.
The reconstruction path IS the causal explanation.

Every f32 value is a BF16 summary + a story of how it got there.
The story is free — it's the tree structure you already built.

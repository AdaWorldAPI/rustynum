# 18 — Three-Plane Dual-Layer Lance Schema

## Ice Cake Layer

> **Prerequisite reading**: 16_concept_pipeline_rethink.md, 17_four_questions_one_architecture.md
> **Does NOT modify**: anything in prompts 01–17, nor existing lance_persistence.rs
> **Produces**: NEW files only. The old flat schema stays untouched until migration runs.

---

## Context (what the previous layers established)

1. **awareness.rs line 167** uses flat XOR `self.superposition.bind(&fp)` — destroys 3D SPO structure
2. **Three separate int8 registers** (S, P, O) replace the flat superposition
3. **Dual representation per plane**: binary 16384-bit for structure + int8 10000D for soaking
4. **Crystallization**: when int8 saturates → `sign()` → binary for permanent storage
5. **σ-2/3 attention mask**: known concepts form a focus lens, not passive storage
6. **Lance already has**: `FixedSizeBinary(2048)` for 16384 bits, `full-zip` O(1) row access
7. **Lance vendor patches**: still commented out (line 313 Cargo.toml), vendor/lance is empty submodule
8. **lance_zero_copy/mod.rs**: `FingerprintBuffer`, `ArrowZeroCopy`, `ScentAwareness` — all work on flat fingerprints

---

## What This Session Builds

### 1. New Arrow Schema: `bind_nodes_v2`

```rust
// file: src/storage/lance_three_plane.rs (NEW FILE)

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, Int8Type};
use std::sync::Arc;

/// Binary structural layer: 16384 bits = 256 × u64 = 2048 bytes
const BINARY_BYTES: i32 = 2048;

/// Int8 soaking layer: 10000 dimensions × 1 byte = 10000 bytes  
const SOAKING_DIM: i32 = 10000;

pub fn bind_nodes_v2_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        // === ADDRESSING ===
        Field::new("addr", DataType::UInt16, false),
        Field::new("label", DataType::Utf8, true),

        // === S PLANE (Subject) ===
        Field::new("s_binary", DataType::FixedSizeBinary(BINARY_BYTES), false),
        Field::new("s_soaking", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int8, false)),
            SOAKING_DIM,
        ), true),  // nullable: only present during active soaking

        // === P PLANE (Predicate) ===
        Field::new("p_binary", DataType::FixedSizeBinary(BINARY_BYTES), false),
        Field::new("p_soaking", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int8, false)),
            SOAKING_DIM,
        ), true),

        // === O PLANE (Object) ===
        Field::new("o_binary", DataType::FixedSizeBinary(BINARY_BYTES), false),
        Field::new("o_soaking", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int8, false)),
            SOAKING_DIM,
        ), true),

        // === COMPOSITE (derived, for backward compat + CAM) ===
        // spo_binary = s_binary ⊗ R_S ⊕ p_binary ⊗ R_P ⊕ o_binary ⊗ R_O
        // Stored redundantly for O(1) CAM lookup without recomputing
        Field::new("spo_binary", DataType::FixedSizeBinary(BINARY_BYTES), true),

        // === SIGMA MASK (which plane is the attention focus) ===
        Field::new("s_sigma", DataType::UInt8, false),  // σ-band per plane
        Field::new("p_sigma", DataType::UInt8, false),
        Field::new("o_sigma", DataType::UInt8, false),

        // === NARS EVIDENCE ===
        Field::new("nars_f", DataType::Float32, false),  // frequency
        Field::new("nars_c", DataType::Float32, false),  // confidence
        Field::new("evidence_count", DataType::UInt32, false),

        // === GATE STATE ===
        // 0=BLOCK, 1=HOLD, 2=FLOW — per plane
        Field::new("s_gate", DataType::UInt8, false),
        Field::new("p_gate", DataType::UInt8, false),
        Field::new("o_gate", DataType::UInt8, false),

        // === ROLE ASSIGNMENT PROVENANCE ===
        // How was S/P/O role decided? Bit flags:
        // 0x01 = grammar position
        // 0x02 = NSM decomposition
        // 0x04 = σ-2/3 context refinement
        // 0x08 = explicit user assignment
        Field::new("role_provenance", DataType::UInt8, false),

        // === METADATA (carried forward from v1) ===
        Field::new("qidx", DataType::UInt8, false),
        Field::new("parent", DataType::UInt16, true),
        Field::new("depth", DataType::UInt8, false),
        Field::new("rung", DataType::UInt8, false),
        Field::new("is_spine", DataType::Boolean, false),
        Field::new("dn_path", DataType::UInt64, true),
        Field::new("payload", DataType::LargeBinary, true),
        Field::new("updated_at", DataType::UInt64, false),
    ])
}
```

### Row Size Budget

```
Per row:
  addr + label overhead                    ≈    20 bytes
  s_binary + p_binary + o_binary           = 6,144 bytes  (3 × 2048)
  s_soaking + p_soaking + o_soaking        = 30,000 bytes (3 × 10000)
  spo_binary (composite)                   = 2,048 bytes
  sigma/gate/nars/provenance/metadata      ≈    50 bytes
  ──────────────────────────────────────────────────────
  TOTAL per row                            ≈ 38,262 bytes (~37 KB)

At 65,536 addressable slots (full BindSpace):
  65,536 × 37 KB = 2.4 GB on disk (uncompressed)

With Lance bit-packing on binary columns:
  Binary columns compress ~2:1 (random bits don't compress much)
  Soaking columns compress ~4:1 (mostly zeros during early soaking)
  Realistic: ~800 MB on disk for full BindSpace
  Hot working set (256 slots): 256 × 37 KB = 9.5 MB (fits L2)
```

### 2. Soaking Column Semantics

The `s_soaking` / `p_soaking` / `o_soaking` columns are **NULLABLE** by design:

```
LIFECYCLE:
  1. Concept first deposited → soaking columns populated, binary = seed
  2. Active soaking → saturating_add per cycle, soaking columns update
  3. Crystallization (gate → FLOW) → sign(soaking) → binary column updates
  4. Post-crystallization → soaking columns SET TO NULL (freed)
  5. Query-time: if soaking IS NULL → concept is crystallized (read binary)
                 if soaking IS NOT NULL → concept is still forming (read soaking)

This means:
  - Crystallized concepts: 6 KB/row (binary only, soaking null)
  - Active concepts: 36 KB/row (both layers)
  - At any moment, maybe 50-200 concepts are actively soaking
  - 200 × 36 KB = 7.2 MB active soaking budget
  - Everything else is 6 KB/row crystallized
```

### 3. Attention Mask Table

Separate from `bind_nodes_v2` because the mask is a GLOBAL structure, not per-node:

```rust
pub fn attention_mask_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        // The mask itself: one 10000D int8 vector
        // = bundle of all σ-2/3 concepts weighted by confidence × recency
        Field::new("mask", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int8, false)),
            SOAKING_DIM,
        ), false),

        // How many concepts contributed to this mask
        Field::new("concept_count", DataType::UInt32, false),

        // Minimum σ-band included (2 = HINT, 3 = KNOWN)
        Field::new("min_sigma", DataType::UInt8, false),

        // Timestamp of last mask rebuild
        Field::new("rebuilt_at", DataType::UInt64, false),

        // Version counter (monotonic, for cache invalidation)
        Field::new("version", DataType::UInt64, false),
    ])
}
```

### 4. Edge Schema Update: `bind_edges_v2`

Edges now carry per-plane fingerprints for the binding:

```rust
pub fn bind_edges_v2_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("from_addr", DataType::UInt16, false),
        Field::new("to_addr", DataType::UInt16, false),
        Field::new("verb_addr", DataType::UInt16, false),

        // Per-plane edge fingerprints (role-bound XOR)
        // edge_s = from.s_binary ⊗ R_AGENT ⊕ to.s_binary ⊗ R_PATIENT
        Field::new("edge_s", DataType::FixedSizeBinary(BINARY_BYTES), false),
        Field::new("edge_p", DataType::FixedSizeBinary(BINARY_BYTES), false),
        Field::new("edge_o", DataType::FixedSizeBinary(BINARY_BYTES), false),

        // Composite (backward compat)
        Field::new("fingerprint", DataType::FixedSizeBinary(BINARY_BYTES), false),

        Field::new("weight", DataType::Float32, false),
        Field::new("nars_f", DataType::Float32, false),
        Field::new("nars_c", DataType::Float32, false),
    ])
}
```

### 5. Migration: v1 → v2

```rust
/// Migration strategy: ADDITIVE, not destructive
///
/// 1. Add new columns to existing tables (Lance schema evolution = free)
/// 2. Backfill: existing `fingerprint` column → copy to all three planes
///    (v1 data has no SPO decomposition, so S=P=O=fingerprint initially)
/// 3. Set spo_binary = fingerprint (identity for v1 data)
/// 4. Set soaking columns = NULL (all v1 data is already "crystallized")
/// 5. Set all sigma = 3 (KNOWN), all gates = FLOW (already committed)
/// 6. Set role_provenance = 0x00 (unknown for legacy data)
/// 7. Keep old `fingerprint` column as `spo_binary` alias

pub struct MigrationV1ToV2;

impl MigrationV1ToV2 {
    /// Zero-copy migration via Lance schema evolution
    ///
    /// Lance supports adding columns without rewriting data.
    /// We add the new columns, then backfill in a single scan.
    pub async fn migrate(dataset_path: &Path) -> Result<()> {
        // Step 1: Open existing v1 dataset
        // Step 2: Add columns via ALTER (Lance DDL)
        //   ADD s_binary  FixedSizeBinary(2048) DEFAULT fingerprint
        //   ADD p_binary  FixedSizeBinary(2048) DEFAULT fingerprint
        //   ADD o_binary  FixedSizeBinary(2048) DEFAULT fingerprint
        //   ADD s_soaking FixedSizeList<Int8>(10000) DEFAULT NULL
        //   ADD p_soaking FixedSizeList<Int8>(10000) DEFAULT NULL
        //   ADD o_soaking FixedSizeList<Int8>(10000) DEFAULT NULL
        //   ADD spo_binary FixedSizeBinary(2048) DEFAULT fingerprint
        //   ADD s_sigma UInt8 DEFAULT 3
        //   ADD p_sigma UInt8 DEFAULT 3
        //   ADD o_sigma UInt8 DEFAULT 3
        //   ADD nars_f Float32 DEFAULT 1.0
        //   ADD nars_c Float32 DEFAULT 0.9
        //   ADD evidence_count UInt32 DEFAULT 1
        //   ADD s_gate UInt8 DEFAULT 2  (FLOW)
        //   ADD p_gate UInt8 DEFAULT 2  (FLOW)
        //   ADD o_gate UInt8 DEFAULT 2  (FLOW)
        //   ADD role_provenance UInt8 DEFAULT 0
        //
        // Step 3: Version bump in bind_state table
        //   schema_version = 2
        //
        // Total disk impact: zero (Lance adds columns as new fragments)
        // Total RAM impact: zero (columns loaded on-demand)
        // Total compute: one scan to verify, no data transformation
        //
        // The old `fingerprint` column is NOT dropped — it becomes
        // the backward-compatible read path. New code reads per-plane.
        // Old code reads `fingerprint` or `spo_binary` (same data).
        todo!("implement when vendor/lance is bootstrapped")
    }
}
```

### 6. Zero-Copy Bridge Update: `ThreePlaneFingerprintBuffer`

```rust
/// Three-plane zero-copy buffer backed by Arrow
///
/// Extends FingerprintBuffer to hold three planes per entry.
/// Each plane has binary (u64 words) and optional soaking (i8 array).
pub struct ThreePlaneFingerprintBuffer {
    /// S plane binary: Arrow Buffer of u64 words
    s_binary: FingerprintBuffer,
    /// P plane binary
    p_binary: FingerprintBuffer,
    /// O plane binary
    o_binary: FingerprintBuffer,

    /// S plane soaking: Option<Arrow Buffer of i8>
    /// None = crystallized (no active soaking)
    s_soaking: Option<SoakingBuffer>,
    /// P plane soaking
    p_soaking: Option<SoakingBuffer>,
    /// O plane soaking
    o_soaking: Option<SoakingBuffer>,

    /// Number of entries
    len: usize,
}

/// Int8 soaking buffer backed by Arrow
pub struct SoakingBuffer {
    buffer: Buffer,  // Arrow buffer, possibly mmap'd
    dim: usize,      // 10000
    len: usize,      // number of entries
}

impl SoakingBuffer {
    /// Get soaking register at index (zero-copy)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[i8]> {
        if index >= self.len { return None; }
        let offset = index * self.dim;
        let ptr = self.buffer.as_ptr() as *const i8;
        unsafe {
            Some(std::slice::from_raw_parts(ptr.add(offset), self.dim))
        }
    }

    /// Deposit evidence: saturating_add into register
    /// REQUIRES MUTABLE ACCESS — only during active soaking
    #[inline]
    pub unsafe fn deposit(&self, index: usize, evidence: &[i8], weight: f32) {
        let offset = index * self.dim;
        let ptr = self.buffer.as_ptr() as *mut i8;
        for i in 0..self.dim {
            let current = *ptr.add(offset + i);
            let contribution = (evidence[i] as f32 * weight).round() as i8;
            *ptr.add(offset + i) = current.saturating_add(contribution);
        }
    }

    /// Evaluate saturation for collapse gate
    pub fn saturation_ratio(&self, index: usize) -> f32 {
        if let Some(register) = self.get(index) {
            let saturated = register.iter()
                .filter(|&&v| v.unsigned_abs() > 100)
                .count();
            saturated as f32 / self.dim as f32
        } else {
            0.0
        }
    }

    /// Crystallize: sign(soaking) → binary fingerprint
    pub fn crystallize(&self, index: usize) -> Option<[u64; 256]> {
        let register = self.get(index)?;
        let mut binary = [0u64; 256];
        for (i, &val) in register.iter().enumerate() {
            // Map 10000 int8 values into 16384 binary bits
            // Strategy: first 10000 bits set by sign, remaining 6384 = 0
            if i < 16384 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                if val > 0 {
                    binary[word_idx] |= 1u64 << bit_idx;
                }
                // val <= 0 → bit stays 0
                // val == 0 → undecided → random tie-break could go here
            }
        }
        Some(binary)
    }
}
```

### 7. Attention Mask Operations

```rust
/// The σ-2/3 attention mask: a single 10000D int8 vector
/// that represents the aggregate focus of all known concepts
pub struct AttentionMask {
    mask: [i8; 10000],
    concept_count: u32,
    min_sigma: u8,
    version: u64,
}

impl AttentionMask {
    /// Rebuild mask from all concepts at σ >= min_sigma
    pub fn rebuild(&mut self, concepts: &ThreePlaneFingerprintBuffer, sigma_levels: &[u8]) {
        self.mask = [0i8; 10000];
        self.concept_count = 0;

        for i in 0..concepts.len {
            if sigma_levels[i] >= self.min_sigma {
                // Get the soaking or derive from binary
                // Weight by NARS confidence
                // saturating_add into mask
                self.concept_count += 1;
            }
        }
        self.version += 1;
    }

    /// Project new concept onto mask → resonance score
    /// Positive = resonance (relates to known)
    /// Near zero = novel (orthogonal to known)
    /// Negative = conflict (contradicts known)
    pub fn project(&self, concept: &[i8; 10000]) -> f32 {
        let mut dot: i64 = 0;
        for i in 0..10000 {
            dot += self.mask[i] as i64 * concept[i] as i64;
        }
        // Normalize by dimension
        dot as f32 / 10000.0
    }

    /// Classify projection result
    pub fn classify(&self, projection: f32) -> AttentionResult {
        if projection > 0.3 {
            AttentionResult::Resonance(projection)
        } else if projection < -0.3 {
            AttentionResult::Conflict(projection)
        } else {
            AttentionResult::Novel(projection)
        }
    }
}

pub enum AttentionResult {
    Resonance(f32),  // relates to known concepts
    Novel(f32),      // genuinely new
    Conflict(f32),   // contradicts known concepts
}
```

---

## File Inventory (what this session creates)

| File | Status | Description |
|------|--------|-------------|
| `src/storage/lance_three_plane.rs` | NEW | Schema defs, ThreePlaneFingerprintBuffer, SoakingBuffer |
| `src/storage/lance_three_plane/attention.rs` | NEW | AttentionMask, projection, classification |
| `src/storage/lance_three_plane/migration.rs` | NEW | V1→V2 additive migration (when vendor/lance ready) |
| `src/storage/lance_three_plane/crystallize.rs` | NEW | int8→binary crystallization, gate evaluation |
| `src/storage/mod.rs` | MODIFY | add `pub mod lance_three_plane;` |

**Does NOT touch:**
- `lance_persistence.rs` (v1 schema stays, v2 is additive)
- `lance_zero_copy/mod.rs` (FingerprintBuffer still works for flat access)
- `awareness.rs` (the flat XOR is still there — wiring comes in a later layer)
- `bind_space.rs` (BindSpace addressing unchanged)
- Cargo.toml vendor patches (still commented out)

---

## Wiring Checklist (for the NEXT ice cake layer, not this one)

```
□ awareness.rs: replace flat XOR with ThreePlaneAwareness
□ cortex.rs: deposit_evidence routes through NSM → role assign → per-plane deposit
□ bind_space.rs: read/write per-plane (s_binary, p_binary, o_binary)
□ Cargo.toml: uncomment lance vendor patches
□ vendor/lance: bootstrap submodule or zipball
□ lance_persistence.rs: run MigrationV1ToV2
□ cam_ops.rs: per-plane CAM addressing
□ satisfaction.rs: per-plane gate → global gate merge
```

---

## Key Invariants

1. **Soaking is transient, binary is permanent.** `s_soaking = NULL` means crystallized.
2. **spo_binary is derived, never primary.** Always recomputable from s/p/o + role vectors.
3. **Migration is additive.** v1 code keeps working — reads `fingerprint`, ignores new columns.
4. **Attention mask is global singleton.** One mask, rebuilt when concepts crystallize.
5. **46KB in L1.** Three planes × 12KB + 10KB mask = 46KB hot working set.
6. **Lance full-zip = O(1) per row.** No decoding neighbors to read one concept.

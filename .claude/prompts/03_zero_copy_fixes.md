# Rustynum Zero-Copy Fixes

**4 targeted changes — rustynum-arrow + rustynum-core**

---

## Fix 1: rustynum-arrow — Deprecate copying path

`arrow_bridge.rs` lines 126-159: `record_batch_to_cogrecords()` does 4× `.to_vec()` per record.
`cogrecord_views()` (lines 205-235) already exists as zero-copy alternative.

```rust
#[deprecated(note = "Use cogrecord_views() for zero-copy")]
pub fn record_batch_to_cogrecords(batch: &RecordBatch) -> Vec<CogRecord> { ... }
```

## Fix 2: rustynum-arrow — lance_io.rs line 52

Current: `records.extend(record_batch_to_cogrecords(&batch))`

Add `read_cogrecord_views()` returning `CogRecordView<'_>` that borrows from the Arrow buffer.

## Fix 3: rustynum-arrow — Deprecate CascadeIndices::build()

`indexed_cascade.rs` lines 83-111: `build()` does 4× `extend_from_slice()`.
`build_from_arrow()` (lines 119-161) already exists as zero-copy alternative.

```rust
#[deprecated(note = "Use build_from_arrow() for zero-copy")]
pub fn build(/* ... */) -> Self { ... }
```

## Fix 4: rustynum-core — Add borrowing CrystalAxis constructor

`spatial_resonance.rs` lines 157-159: `from_flat_bytes()` does 3× `.to_vec()` because `CrystalAxis::from_bf16_bytes()` takes `Vec<u8>`.

Add:
```rust
impl CrystalAxis {
    /// Zero-copy: borrows from caller
    pub fn from_bf16_slice(data: &[u8]) -> Self { /* borrow instead of .to_vec() */ }
    
    // Keep existing for owned use:
    // pub fn from_bf16_bytes(data: Vec<u8>) -> Self { ... }
}
```

---

## Verification

```bash
cargo test --package rustynum-arrow
cargo test --package rustynum-core -- spatial_resonance
# Deprecation warnings should appear for old paths
```

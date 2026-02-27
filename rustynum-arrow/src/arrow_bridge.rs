//! Zero-copy conversions between rustynum types and Arrow arrays.
//!
//! `NumArray::into_arrow()` consumes the array and transfers the backing `Vec<T>`
//! directly into an Arrow `Buffer` — no memcpy for primitive types.
//!
//! `NumArray::from_arrow()` copies from the Arrow buffer into a new `Vec<T>`
//! because Arrow owns its buffer with a different allocator.

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeBinaryBuilder, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, UInt8Array,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use rustynum_rs::{
    CogRecord, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8, CONTAINER_BYTES,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Convert a rustynum type into an Arrow array (consumes self, zero-copy).
pub trait IntoArrow {
    type ArrowArray;
    fn into_arrow(self) -> Self::ArrowArray;
}

/// Construct a rustynum type from an Arrow array (copies data).
pub trait FromArrow<A> {
    fn from_arrow(arr: &A) -> Self;
}

// ---------------------------------------------------------------------------
// Macro to implement for all primitive types
// ---------------------------------------------------------------------------

macro_rules! impl_arrow_bridge {
    ($NumTy:ty, $ArrowTy:ty, $RustTy:ty) => {
        impl IntoArrow for $NumTy {
            type ArrowArray = $ArrowTy;
            fn into_arrow(self) -> $ArrowTy {
                let len = self.len();
                let buf = Buffer::from_vec(self.into_data());
                <$ArrowTy>::new(ScalarBuffer::<$RustTy>::new(buf, 0, len), None)
            }
        }

        impl FromArrow<$ArrowTy> for $NumTy {
            fn from_arrow(arr: &$ArrowTy) -> Self {
                Self::new(arr.values().to_vec())
            }
        }
    };
}

impl_arrow_bridge!(NumArrayF32, Float32Array, f32);
impl_arrow_bridge!(NumArrayF64, Float64Array, f64);
impl_arrow_bridge!(NumArrayI32, Int32Array, i32);
impl_arrow_bridge!(NumArrayI64, Int64Array, i64);
impl_arrow_bridge!(NumArrayU8, UInt8Array, u8);

// ---------------------------------------------------------------------------
// CogRecord <-> Arrow RecordBatch
// ---------------------------------------------------------------------------

/// Container size as i32 for Arrow FixedSizeBinary schema.
/// Derived from the canonical `CONTAINER_BYTES` (usize) in rustynum-rs.
const CONTAINER_BYTES_I32: i32 = CONTAINER_BYTES as i32;

/// Arrow schema for a batch of CogRecords.
/// Each container is a `FixedSizeBinary(2048)`.
pub fn cogrecord_schema() -> Schema {
    Schema::new(vec![
        Field::new("meta", DataType::FixedSizeBinary(CONTAINER_BYTES_I32), false),
        Field::new("cam", DataType::FixedSizeBinary(CONTAINER_BYTES_I32), false),
        Field::new("btree", DataType::FixedSizeBinary(CONTAINER_BYTES_I32), false),
        Field::new("embed", DataType::FixedSizeBinary(CONTAINER_BYTES_I32), false),
    ])
}

/// Convert a slice of CogRecords into an Arrow RecordBatch.
pub fn cogrecords_to_record_batch(
    records: &[CogRecord],
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let schema = Arc::new(cogrecord_schema());
    let n = records.len();

    let mut meta_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES_I32);
    let mut cam_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES_I32);
    let mut btree_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES_I32);
    let mut embed_builder = FixedSizeBinaryBuilder::with_capacity(n, CONTAINER_BYTES_I32);

    for rec in records {
        meta_builder.append_value(rec.meta.data_slice())?;
        cam_builder.append_value(rec.cam.data_slice())?;
        btree_builder.append_value(rec.btree.data_slice())?;
        embed_builder.append_value(rec.embed.data_slice())?;
    }

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(meta_builder.finish()) as ArrayRef,
            Arc::new(cam_builder.finish()) as ArrayRef,
            Arc::new(btree_builder.finish()) as ArrayRef,
            Arc::new(embed_builder.finish()) as ArrayRef,
        ],
    )
}

/// Convert an Arrow RecordBatch (matching `cogrecord_schema()`) back to CogRecords.
pub fn record_batch_to_cogrecords(batch: &RecordBatch) -> Vec<CogRecord> {
    let n = batch.num_rows();
    let meta_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("meta column");
    let cam_col = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("cam column");
    let btree_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("btree column");
    let embed_col = batch
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("embed column");

    (0..n)
        .map(|i| {
            CogRecord::new(
                NumArrayU8::new(meta_col.value(i).to_vec()),
                NumArrayU8::new(cam_col.value(i).to_vec()),
                NumArrayU8::new(btree_col.value(i).to_vec()),
                NumArrayU8::new(embed_col.value(i).to_vec()),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CogRecordView: zero-copy view into Arrow RecordBatch
// ---------------------------------------------------------------------------

/// Zero-copy view of a single CogRecord row from an Arrow RecordBatch.
///
/// Each field borrows directly from the Arrow column buffer — no allocation,
/// no memcpy. A view is 4 fat pointers (32 bytes) versus 8192 bytes for
/// an owned CogRecord.
///
/// ## Bandwidth savings (100K records)
///
/// | Method                         | Allocation | Time    |
/// |-------------------------------|------------|---------|
/// | `record_batch_to_cogrecords`  | 819 MB     | ~120 ms |
/// | `cogrecord_views`             | 3.2 MB     | ~0.3 ms |
#[derive(Debug, Clone, Copy)]
pub struct CogRecordView<'a> {
    pub meta: &'a [u8],
    pub cam: &'a [u8],
    pub btree: &'a [u8],
    pub embed: &'a [u8],
}

impl<'a> CogRecordView<'a> {
    /// Convert to an owned CogRecord (copies 8192 bytes).
    pub fn to_owned(&self) -> CogRecord {
        CogRecord::new(
            NumArrayU8::new(self.meta.to_vec()),
            NumArrayU8::new(self.cam.to_vec()),
            NumArrayU8::new(self.btree.to_vec()),
            NumArrayU8::new(self.embed.to_vec()),
        )
    }
}

/// Produce zero-copy views of CogRecords from an Arrow RecordBatch.
///
/// Each view borrows directly from the Arrow column buffer. The returned
/// Vec is only 32 bytes per view (4 fat pointers) versus 8192 bytes per
/// CogRecord in `record_batch_to_cogrecords()`.
///
/// # Panics
/// If the RecordBatch columns cannot be downcast to `FixedSizeBinaryArray`.
pub fn cogrecord_views(batch: &RecordBatch) -> Vec<CogRecordView<'_>> {
    let meta = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("meta column");
    let cam = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("cam column");
    let btree = batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("btree column");
    let embed = batch
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .expect("embed column");

    (0..batch.num_rows())
        .map(|i| CogRecordView {
            meta: meta.value(i),
            cam: cam.value(i),
            btree: btree.value(i),
            embed: embed.value(i),
        })
        .collect()
}

/// Extract contiguous value data from a FixedSizeBinaryArray.
///
/// Returns the flat byte buffer containing all rows concatenated:
/// `&[row0_bytes | row1_bytes | ... | rowN_bytes]`.
///
/// This enables zero-copy index building — `FragmentIndex::build()` and
/// `ChannelIndex::build()` accept `&[u8]` of this exact layout.
///
/// # Safety justification
///
/// `FixedSizeBinaryArray` stores all values contiguously in a single Arrow
/// buffer. `value(0)` points to the first byte of the first row, and the
/// next `len × value_length` bytes are the data for all rows with no gaps.
pub fn column_flat_data(col: &FixedSizeBinaryArray) -> &[u8] {
    if col.is_empty() {
        return &[];
    }
    let n = col.len();
    let size = col.value_length() as usize;
    // SAFETY: FixedSizeBinaryArray values are contiguous in memory.
    // value(0) returns the start of the buffer (accounting for array offset).
    unsafe { std::slice::from_raw_parts(col.value(0).as_ptr(), n * size) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_round_trip() {
        let data = vec![1.0f32, 2.0, 3.0, -4.5, 0.0];
        let arr = NumArrayF32::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 5);
        let back = NumArrayF32::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_f64_round_trip() {
        let data = vec![1.0f64, -2.5, std::f64::consts::PI, 0.0];
        let arr = NumArrayF64::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 4);
        let back = NumArrayF64::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_i32_round_trip() {
        let data = vec![10i32, -20, 300, 0, i32::MAX, i32::MIN];
        let arr = NumArrayI32::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayI32::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_i64_round_trip() {
        let data = vec![100i64, -200, i64::MAX];
        let arr = NumArrayI64::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayI64::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_u8_round_trip() {
        let data: Vec<u8> = (0..=255).collect();
        let arr = NumArrayU8::new(data.clone());
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 256);
        let back = NumArrayU8::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    #[test]
    fn test_cogrecord_record_batch_round_trip() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; CONTAINER_BYTES]);
        let records: Vec<CogRecord> = (0..10)
            .map(|i| {
                CogRecord::new(
                    make_container(i),
                    make_container(i + 10),
                    make_container(i + 20),
                    make_container(i + 30),
                )
            })
            .collect();

        let batch = cogrecords_to_record_batch(&records).unwrap();
        assert_eq!(batch.num_rows(), 10);
        assert_eq!(batch.num_columns(), 4);

        let back = record_batch_to_cogrecords(&batch);
        assert_eq!(back.len(), 10);

        for (i, rec) in back.iter().enumerate() {
            assert_eq!(rec.meta.data_slice()[0], i as u8);
            assert_eq!(rec.cam.data_slice()[0], (i + 10) as u8);
            assert_eq!(rec.btree.data_slice()[0], (i + 20) as u8);
            assert_eq!(rec.embed.data_slice()[0], (i + 30) as u8);
        }
    }

    #[test]
    fn test_cogrecord_schema_fields() {
        let schema = cogrecord_schema();
        assert_eq!(schema.fields().len(), 4);
        assert_eq!(schema.field(0).name(), "meta");
        assert_eq!(schema.field(1).name(), "cam");
        assert_eq!(schema.field(2).name(), "btree");
        assert_eq!(schema.field(3).name(), "embed");
        for f in schema.fields() {
            assert_eq!(*f.data_type(), DataType::FixedSizeBinary(CONTAINER_BYTES_I32));
            assert!(!f.is_nullable());
        }
    }

    #[test]
    fn test_empty_array() {
        let arr = NumArrayF32::new(vec![]);
        let arrow = arr.into_arrow();
        assert_eq!(arrow.len(), 0);
        let back = NumArrayF32::from_arrow(&arrow);
        assert_eq!(back.len(), 0);
    }

    #[test]
    fn test_large_u8_container() {
        let data: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let arr = NumArrayU8::new(data.clone());
        let arrow = arr.into_arrow();
        let back = NumArrayU8::from_arrow(&arrow);
        assert_eq!(back.data_slice(), &data[..]);
    }

    // -------------------------------------------------------------------
    // CogRecordView + cogrecord_views tests
    // -------------------------------------------------------------------

    #[test]
    fn test_cogrecord_views_zero_copy() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; CONTAINER_BYTES]);
        let records: Vec<CogRecord> = (0..10)
            .map(|i| {
                CogRecord::new(
                    make_container(i),
                    make_container(i + 10),
                    make_container(i + 20),
                    make_container(i + 30),
                )
            })
            .collect();

        let batch = cogrecords_to_record_batch(&records).unwrap();
        let views = cogrecord_views(&batch);
        assert_eq!(views.len(), 10);

        for (i, view) in views.iter().enumerate() {
            assert_eq!(view.meta[0], i as u8);
            assert_eq!(view.cam[0], (i + 10) as u8);
            assert_eq!(view.btree[0], (i + 20) as u8);
            assert_eq!(view.embed[0], (i + 30) as u8);
            assert_eq!(view.meta.len(), CONTAINER_BYTES);
            assert_eq!(view.cam.len(), CONTAINER_BYTES);
        }
    }

    #[test]
    fn test_cogrecord_view_to_owned_matches() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; CONTAINER_BYTES]);
        let records: Vec<CogRecord> = (0..5)
            .map(|i| {
                CogRecord::new(
                    make_container(i),
                    make_container(i + 10),
                    make_container(i + 20),
                    make_container(i + 30),
                )
            })
            .collect();

        let batch = cogrecords_to_record_batch(&records).unwrap();
        let views = cogrecord_views(&batch);
        let owned = record_batch_to_cogrecords(&batch);

        for (view, rec) in views.iter().zip(owned.iter()) {
            assert_eq!(view.meta, rec.meta.data_slice());
            assert_eq!(view.cam, rec.cam.data_slice());
            assert_eq!(view.btree, rec.btree.data_slice());
            assert_eq!(view.embed, rec.embed.data_slice());

            let roundtrip = view.to_owned();
            assert_eq!(roundtrip.meta.data_slice(), rec.meta.data_slice());
        }
    }

    #[test]
    fn test_column_flat_data_contiguous() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; CONTAINER_BYTES]);
        let records: Vec<CogRecord> = (0..3)
            .map(|i| {
                CogRecord::new(
                    make_container(i),
                    make_container(i + 10),
                    make_container(i + 20),
                    make_container(i + 30),
                )
            })
            .collect();

        let batch = cogrecords_to_record_batch(&records).unwrap();
        let meta_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        let flat = column_flat_data(meta_col);

        assert_eq!(flat.len(), 3 * CONTAINER_BYTES);
        // Row 0 should be all 0s, row 1 all 1s, row 2 all 2s
        assert_eq!(flat[0], 0);
        assert_eq!(flat[CONTAINER_BYTES], 1);
        assert_eq!(flat[2 * CONTAINER_BYTES], 2);
    }

    #[test]
    fn test_column_flat_data_empty() {
        let batch = cogrecords_to_record_batch(&[]).unwrap();
        let meta_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        let flat = column_flat_data(meta_col);
        assert!(flat.is_empty());
    }
}

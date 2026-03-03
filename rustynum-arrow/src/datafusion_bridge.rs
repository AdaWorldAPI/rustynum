//! DataFusion bridge: zero-copy cascade scan over Arrow `FixedSizeBinaryArray` columns.
//!
//! When DataFusion scans a Lance dataset, it returns `RecordBatch` with
//! `FixedSizeBinaryArray` columns for each CogRecord container. This module
//! provides zero-copy access to those columns for SIMD-accelerated Hamming
//! search — no 16KB-per-record allocation.

use arrow::array::{Array, FixedSizeBinaryArray};
use rustynum_core::simd::hamming_distance;
use rustynum_rs::CogRecord;

/// Zero-copy slice into an Arrow `FixedSizeBinaryArray`'s backing buffer.
///
/// Arrow's `FixedSizeBinaryArray` stores values contiguously in memory,
/// which is the exact layout `hamming_search_adaptive` expects for a
/// packed database. This function returns the raw byte slice — no copy.
///
/// # Safety
///
/// The returned slice borrows from the `FixedSizeBinaryArray`. The caller
/// must ensure the array outlives the slice.
pub fn arrow_to_flat_bytes(col: &FixedSizeBinaryArray) -> &[u8] {
    col.value_data()
}

/// Scan an Arrow `FixedSizeBinaryArray` column with Hamming distance,
/// returning indices and distances of candidates below `max_distance`.
///
/// This is the zero-copy path: the Arrow column's backing buffer is used
/// directly as the packed database for linear Hamming scan. No allocation
/// for the database itself.
pub fn hamming_scan_column(
    query: &[u8],
    column: &FixedSizeBinaryArray,
    max_distance: u64,
) -> Vec<(usize, u64)> {
    let n = column.len();
    let vec_len = column.value_length() as usize;
    let flat = arrow_to_flat_bytes(column);
    let mut results = Vec::new();

    for i in 0..n {
        let offset = i * vec_len;
        let candidate = &flat[offset..offset + vec_len];

        // Scalar Hamming via u64 popcount
        let dist = hamming_distance(query, candidate);
        if dist <= max_distance {
            results.push((i, dist));
        }
    }

    results
}

/// 4-channel cascade scan over a RecordBatch of CogRecords.
///
/// Scans all 4 container columns with per-channel thresholds.
/// Uses compound early exit: if META exceeds threshold, skip CAM/BTREE/EMBED.
///
/// Zero-copy: operates directly on Arrow column buffers.
pub fn cascade_scan_4ch(
    query: &CogRecord,
    meta_col: &FixedSizeBinaryArray,
    cam_col: &FixedSizeBinaryArray,
    btree_col: &FixedSizeBinaryArray,
    embed_col: &FixedSizeBinaryArray,
    thresholds: [u64; 4],
) -> Vec<(usize, [u64; 4])> {
    let n = meta_col.len();
    assert_eq!(
        cam_col.len(),
        n,
        "cascade_scan_4ch: cam_col length {} != meta_col length {}",
        cam_col.len(),
        n
    );
    assert_eq!(
        btree_col.len(),
        n,
        "cascade_scan_4ch: btree_col length {} != meta_col length {}",
        btree_col.len(),
        n
    );
    assert_eq!(
        embed_col.len(),
        n,
        "cascade_scan_4ch: embed_col length {} != meta_col length {}",
        embed_col.len(),
        n
    );
    let vec_len = meta_col.value_length() as usize;
    let meta_flat = arrow_to_flat_bytes(meta_col);
    let cam_flat = arrow_to_flat_bytes(cam_col);
    let btree_flat = arrow_to_flat_bytes(btree_col);
    let embed_flat = arrow_to_flat_bytes(embed_col);

    let q_meta = query.meta.data_slice();
    let q_cam = query.cam.data_slice();
    let q_btree = query.btree.data_slice();
    let q_embed = query.embed.data_slice();

    let mut results = Vec::new();

    for i in 0..n {
        let offset = i * vec_len;

        // Stage 1: META (cheapest rejection)
        let meta_dist = hamming_distance(q_meta, &meta_flat[offset..offset + vec_len]);
        if meta_dist > thresholds[0] {
            continue;
        }

        // Stage 2: CAM
        let cam_dist = hamming_distance(q_cam, &cam_flat[offset..offset + vec_len]);
        if cam_dist > thresholds[1] {
            continue;
        }

        // Stage 3: BTREE
        let btree_dist = hamming_distance(q_btree, &btree_flat[offset..offset + vec_len]);
        if btree_dist > thresholds[2] {
            continue;
        }

        // Stage 4: EMBED
        let embed_dist = hamming_distance(q_embed, &embed_flat[offset..offset + vec_len]);
        if embed_dist > thresholds[3] {
            continue;
        }

        results.push((i, [meta_dist, cam_dist, btree_dist, embed_dist]));
    }

    results
}

// ---------------------------------------------------------------------------
// DataFusion UDFs — SIMD-accelerated scalar functions for SQL queries
// ---------------------------------------------------------------------------
//
// These wrap rustynum-core's SIMD kernels as DataFusion ScalarUDFs,
// so they can be used in SQL: `SELECT rusty_hamming(a.fp, b.fp) FROM ...`
//
// Broadcast semantics:
//   Array × Array  → pairwise (vectorized batch)
//   Array × Scalar → broadcast search (common case)
//   Scalar × Scalar → single compute

#[cfg(feature = "datafusion")]
pub mod udfs {
    use std::any::Any;
    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, BinaryArray, FixedSizeBinaryArray, Float32Array, UInt64Array,
    };
    use arrow::datatypes::DataType;
    use datafusion::common::Result;
    use datafusion::logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, TypeSignature,
        Volatility,
    };
    use rustynum_core::simd::{hamming_distance, popcount};

    // ── helpers ──────────────────────────────────────────────────────────

    fn downcast_binary(arr: &ArrayRef) -> Result<Vec<Option<&[u8]>>> {
        if let Some(fsb) = arr.as_any().downcast_ref::<FixedSizeBinaryArray>() {
            Ok((0..fsb.len())
                .map(|i| if fsb.is_null(i) { None } else { Some(fsb.value(i)) })
                .collect())
        } else if let Some(bin) = arr.as_any().downcast_ref::<BinaryArray>() {
            Ok((0..bin.len())
                .map(|i| if bin.is_null(i) { None } else { Some(bin.value(i)) })
                .collect())
        } else {
            Err(datafusion::error::DataFusionError::Execution(format!(
                "expected Binary or FixedSizeBinary array, got {:?}",
                arr.data_type()
            )))
        }
    }

    fn expand_to_arrays(a: &ColumnarValue, b: &ColumnarValue) -> Result<(ArrayRef, ArrayRef)> {
        match (a, b) {
            (ColumnarValue::Array(a), ColumnarValue::Array(b)) => Ok((a.clone(), b.clone())),
            (ColumnarValue::Array(a), ColumnarValue::Scalar(sb)) => {
                Ok((a.clone(), sb.to_array_of_size(a.len())?))
            }
            (ColumnarValue::Scalar(sa), ColumnarValue::Array(b)) => {
                Ok((sa.to_array_of_size(b.len())?, b.clone()))
            }
            (ColumnarValue::Scalar(sa), ColumnarValue::Scalar(sb)) => {
                Ok((sa.to_array_of_size(1)?, sb.to_array_of_size(1)?))
            }
        }
    }

    fn binary_pair_signature() -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Binary, DataType::Binary]),
                TypeSignature::Exact(vec![DataType::LargeBinary, DataType::LargeBinary]),
                TypeSignature::Any(2),
            ],
            Volatility::Immutable,
        )
    }

    fn unary_binary_signature() -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Binary]),
                TypeSignature::Exact(vec![DataType::LargeBinary]),
                TypeSignature::Any(1),
            ],
            Volatility::Immutable,
        )
    }

    // ═══════════════════════════════════════════════════════════════════
    // 1. rusty_hamming(a, b) → UInt64
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct RustyHammingUdf {
        signature: Signature,
    }

    impl RustyHammingUdf {
        pub fn new() -> Self {
            Self { signature: binary_pair_signature() }
        }
    }

    impl ScalarUDFImpl for RustyHammingUdf {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "rusty_hamming" }
        fn signature(&self) -> &Signature { &self.signature }
        fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt64) }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let (a_arr, b_arr) = expand_to_arrays(&args.args[0], &args.args[1])?;
            let a_vals = downcast_binary(&a_arr)?;
            let b_vals = downcast_binary(&b_arr)?;
            let results: UInt64Array = a_vals
                .iter()
                .zip(b_vals.iter())
                .map(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => Some(hamming_distance(a, b)),
                    _ => None,
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(results)))
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. rusty_similarity(a, b) → Float32   [1.0 - hamming/(len*8)]
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct RustySimilarityUdf {
        signature: Signature,
    }

    impl RustySimilarityUdf {
        pub fn new() -> Self {
            Self { signature: binary_pair_signature() }
        }
    }

    impl ScalarUDFImpl for RustySimilarityUdf {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "rusty_similarity" }
        fn signature(&self) -> &Signature { &self.signature }
        fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::Float32) }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let (a_arr, b_arr) = expand_to_arrays(&args.args[0], &args.args[1])?;
            let a_vals = downcast_binary(&a_arr)?;
            let b_vals = downcast_binary(&b_arr)?;
            let results: Float32Array = a_vals
                .iter()
                .zip(b_vals.iter())
                .map(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => {
                        let total_bits = (a.len().min(b.len()) * 8) as f32;
                        if total_bits == 0.0 {
                            Some(1.0f32)
                        } else {
                            Some(1.0 - hamming_distance(a, b) as f32 / total_bits)
                        }
                    }
                    _ => None,
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(results)))
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. rusty_popcount(x) → UInt64
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct RustyPopcountUdf {
        signature: Signature,
    }

    impl RustyPopcountUdf {
        pub fn new() -> Self {
            Self { signature: unary_binary_signature() }
        }
    }

    impl ScalarUDFImpl for RustyPopcountUdf {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "rusty_popcount" }
        fn signature(&self) -> &Signature { &self.signature }
        fn return_type(&self, _: &[DataType]) -> Result<DataType> { Ok(DataType::UInt64) }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let arr = match &args.args[0] {
                ColumnarValue::Array(a) => a.clone(),
                ColumnarValue::Scalar(s) => s.to_array_of_size(1)?,
            };
            let vals = downcast_binary(&arr)?;
            let results: UInt64Array = vals
                .iter()
                .map(|v| v.map(popcount))
                .collect();
            Ok(ColumnarValue::Array(Arc::new(results)))
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. rusty_xor_bind(a, b) → Binary  [element-wise XOR for VSA/HDC]
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct RustyXorBindUdf {
        signature: Signature,
    }

    impl RustyXorBindUdf {
        pub fn new() -> Self {
            Self { signature: binary_pair_signature() }
        }
    }

    impl ScalarUDFImpl for RustyXorBindUdf {
        fn as_any(&self) -> &dyn Any { self }
        fn name(&self) -> &str { "rusty_xor_bind" }
        fn signature(&self) -> &Signature { &self.signature }
        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            match &arg_types[0] {
                dt @ DataType::FixedSizeBinary(_) => Ok(dt.clone()),
                _ => Ok(DataType::Binary),
            }
        }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let (a_arr, b_arr) = expand_to_arrays(&args.args[0], &args.args[1])?;
            let a_vals = downcast_binary(&a_arr)?;
            let b_vals = downcast_binary(&b_arr)?;
            let results: BinaryArray = a_vals
                .iter()
                .zip(b_vals.iter())
                .map(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => {
                        let len = a.len().min(b.len());
                        let xored: Vec<u8> = (0..len).map(|i| a[i] ^ b[i]).collect();
                        Some(xored)
                    }
                    _ => None,
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(results)))
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Registration
    // ═══════════════════════════════════════════════════════════════════

    /// Register all rustynum SIMD-accelerated UDFs with a DataFusion session.
    pub fn register_rustynum_udfs(ctx: &datafusion::execution::context::SessionContext) {
        ctx.register_udf(ScalarUDF::from(RustyHammingUdf::new()));
        ctx.register_udf(ScalarUDF::from(RustySimilarityUdf::new()));
        ctx.register_udf(ScalarUDF::from(RustyPopcountUdf::new()));
        ctx.register_udf(ScalarUDF::from(RustyXorBindUdf::new()));
    }

    /// Return all rustynum UDFs as a Vec for custom registration.
    pub fn all_rustynum_udfs() -> Vec<ScalarUDF> {
        vec![
            ScalarUDF::from(RustyHammingUdf::new()),
            ScalarUDF::from(RustySimilarityUdf::new()),
            ScalarUDF::from(RustyPopcountUdf::new()),
            ScalarUDF::from(RustyXorBindUdf::new()),
        ]
    }
}

#[cfg(feature = "datafusion")]
pub use udfs::{
    all_rustynum_udfs, register_rustynum_udfs, RustyHammingUdf, RustyPopcountUdf,
    RustySimilarityUdf, RustyXorBindUdf,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::FixedSizeBinaryBuilder;
    use rustynum_rs::{NumArrayU8, CONTAINER_BYTES};

    fn make_column(data: &[&[u8]], element_size: i32) -> FixedSizeBinaryArray {
        let mut builder = FixedSizeBinaryBuilder::with_capacity(data.len(), element_size);
        for row in data {
            builder
                .append_value(row)
                .expect("append_value failed: row length != element_size");
        }
        builder.finish()
    }

    #[test]
    fn test_arrow_to_flat_bytes_contiguous() {
        let row0 = vec![0u8; CONTAINER_BYTES];
        let row1 = vec![1u8; CONTAINER_BYTES];
        let col = make_column(&[&row0, &row1], CONTAINER_BYTES as i32);

        let flat = arrow_to_flat_bytes(&col);
        assert_eq!(flat.len(), CONTAINER_BYTES * 2);
        assert_eq!(flat[0], 0);
        assert_eq!(flat[CONTAINER_BYTES], 1);
    }

    #[test]
    fn test_hamming_scan_column_exact_match() {
        let query = vec![0xAAu8; CONTAINER_BYTES];
        let rows: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                if i == 5 {
                    vec![0xAAu8; CONTAINER_BYTES] // exact match
                } else {
                    vec![i as u8; CONTAINER_BYTES]
                }
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let col = make_column(&refs, CONTAINER_BYTES as i32);

        let results = hamming_scan_column(&query, &col, 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 5);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_hamming_scan_column_threshold() {
        let query = vec![0u8; CONTAINER_BYTES];
        let close = vec![1u8; CONTAINER_BYTES]; // Hamming distance = CONTAINER_BYTES * 1 bit
        let far = vec![0xFFu8; CONTAINER_BYTES]; // Hamming distance = CONTAINER_BYTES * 8
        let col = make_column(&[&close, &far], CONTAINER_BYTES as i32);

        // close: each byte=0x01 vs 0x00 → 1 bit per byte → CONTAINER_BYTES bits total
        // far: each byte=0xFF vs 0x00 → 8 bits per byte → CONTAINER_BYTES * 8 bits
        // Threshold must be above close distance but below far distance
        let threshold = (CONTAINER_BYTES as u64) + 1000;
        let results = hamming_scan_column(&query, &col, threshold);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_cascade_scan_4ch() {
        let make_container = |val: u8| NumArrayU8::new(vec![val; CONTAINER_BYTES]);
        let query = CogRecord::new(
            make_container(0),
            make_container(0),
            make_container(0),
            make_container(0),
        );

        // Create 5 records: record 2 is close to query, others are far
        let records: Vec<CogRecord> = (0..5)
            .map(|i| {
                if i == 2 {
                    CogRecord::new(
                        make_container(1), // close in META
                        make_container(1), // close in CAM
                        make_container(1), // close in BTREE
                        make_container(1), // close in EMBED
                    )
                } else {
                    CogRecord::new(
                        make_container(0xFF),
                        make_container(0xFF),
                        make_container(0xFF),
                        make_container(0xFF),
                    )
                }
            })
            .collect();

        let meta_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.meta.data_slice().to_vec())
            .collect();
        let cam_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.cam.data_slice().to_vec())
            .collect();
        let btree_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.btree.data_slice().to_vec())
            .collect();
        let embed_rows: Vec<Vec<u8>> = records
            .iter()
            .map(|r| r.embed.data_slice().to_vec())
            .collect();

        let m_refs: Vec<&[u8]> = meta_rows.iter().map(|r| r.as_slice()).collect();
        let c_refs: Vec<&[u8]> = cam_rows.iter().map(|r| r.as_slice()).collect();
        let b_refs: Vec<&[u8]> = btree_rows.iter().map(|r| r.as_slice()).collect();
        let e_refs: Vec<&[u8]> = embed_rows.iter().map(|r| r.as_slice()).collect();

        let meta_col = make_column(&m_refs, CONTAINER_BYTES as i32);
        let cam_col = make_column(&c_refs, CONTAINER_BYTES as i32);
        let btree_col = make_column(&b_refs, CONTAINER_BYTES as i32);
        let embed_col = make_column(&e_refs, CONTAINER_BYTES as i32);

        // Close record has Hamming = CONTAINER_BYTES (1 bit per byte), far = CONTAINER_BYTES * 8
        // Threshold must pass close but reject far
        let threshold = (CONTAINER_BYTES as u64) + 1000;
        let results = cascade_scan_4ch(
            &query,
            &meta_col,
            &cam_col,
            &btree_col,
            &embed_col,
            [threshold, threshold, threshold, threshold],
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_hamming_distance_correctness() {
        let a = vec![0u8; 64];
        let b = vec![0xFFu8; 64];
        assert_eq!(hamming_distance(&a, &b), 64 * 8);

        let c = vec![0u8; 64];
        assert_eq!(hamming_distance(&a, &c), 0);
    }
}

// ---------------------------------------------------------------------------
// DataFusion UDF tests (require `datafusion` feature)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "datafusion"))]
mod udf_tests {
    use std::sync::Arc;

    use arrow::array::{BinaryArray, FixedSizeBinaryBuilder};
    use arrow::datatypes::{DataType, Field};
    use datafusion::config::ConfigOptions;
    use datafusion::execution::context::SessionContext;
    use datafusion::execution::FunctionRegistry;
    use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl};

    use super::udfs::*;

    fn make_fsb_column(data: &[&[u8]], element_size: i32) -> Arc<dyn arrow::array::Array> {
        let mut builder =
            FixedSizeBinaryBuilder::with_capacity(data.len(), element_size);
        for row in data {
            builder.append_value(row).unwrap();
        }
        Arc::new(builder.finish())
    }

    fn make_args(args: Vec<ColumnarValue>, num_rows: usize, ret_dt: DataType) -> ScalarFunctionArgs {
        let arg_fields: Vec<_> = args
            .iter()
            .enumerate()
            .map(|(i, cv)| {
                let dt = cv.data_type();
                Arc::new(Field::new(format!("c{i}"), dt, true))
            })
            .collect();
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: num_rows,
            return_field: Arc::new(Field::new("out", ret_dt, true)),
            config_options: Arc::new(ConfigOptions::default()),
        }
    }

    #[test]
    fn test_udf_hamming_array_x_array() {
        let a = make_fsb_column(&[&[0u8; 8], &[0xFFu8; 8]], 8);
        let b = make_fsb_column(&[&[0u8; 8], &[0u8; 8]], 8);

        let udf = RustyHammingUdf::new();
        let result = udf
            .invoke_with_args(make_args(
                vec![ColumnarValue::Array(a), ColumnarValue::Array(b)],
                2,
                DataType::UInt64,
            ))
            .unwrap();

        if let ColumnarValue::Array(arr) = result {
            let u64arr = arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            assert_eq!(u64arr.value(0), 0); // 0x00 vs 0x00 → 0
            assert_eq!(u64arr.value(1), 64); // 0xFF vs 0x00 → 8 bits * 8 bytes
        } else {
            panic!("expected Array result");
        }
    }

    #[test]
    fn test_udf_similarity_normalized() {
        let a = make_fsb_column(&[&[0u8; 8]], 8);
        let b = make_fsb_column(&[&[0xFFu8; 8]], 8);

        let udf = RustySimilarityUdf::new();
        let result = udf
            .invoke_with_args(make_args(
                vec![ColumnarValue::Array(a), ColumnarValue::Array(b)],
                1,
                DataType::Float32,
            ))
            .unwrap();

        if let ColumnarValue::Array(arr) = result {
            let f32arr = arr
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .unwrap();
            assert!((f32arr.value(0) - 0.0).abs() < 0.001); // max distance → 0.0 similarity
        } else {
            panic!("expected Array result");
        }
    }

    #[test]
    fn test_udf_popcount() {
        let udf = RustyPopcountUdf::new();
        let input = make_fsb_column(&[&[0xFFu8; 4]], 4);
        let result = udf
            .invoke_with_args(make_args(
                vec![ColumnarValue::Array(input)],
                1,
                DataType::UInt64,
            ))
            .unwrap();

        if let ColumnarValue::Array(arr) = result {
            let u64arr = arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap();
            assert_eq!(u64arr.value(0), 32); // 4 bytes * 8 bits
        } else {
            panic!("expected Array result");
        }
    }

    #[test]
    fn test_udf_xor_bind_self_inverse() {
        let a_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let b_data = vec![0x11, 0x22, 0x33, 0x44];
        let a = make_fsb_column(&[&a_data], 4);
        let b = make_fsb_column(&[&b_data], 4);

        let xor_udf = RustyXorBindUdf::new();

        // XOR a and b
        let result = xor_udf
            .invoke_with_args(make_args(
                vec![ColumnarValue::Array(a.clone()), ColumnarValue::Array(b.clone())],
                1,
                DataType::Binary,
            ))
            .unwrap();

        // XOR result with b → should recover a (XOR is self-inverse)
        let result2 = xor_udf
            .invoke_with_args(make_args(
                vec![result, ColumnarValue::Array(b)],
                1,
                DataType::Binary,
            ))
            .unwrap();

        if let ColumnarValue::Array(arr) = result2 {
            let bin = arr.as_any().downcast_ref::<BinaryArray>().unwrap();
            assert_eq!(bin.value(0), &a_data);
        } else {
            panic!("expected Array result");
        }
    }

    #[test]
    fn test_register_rustynum_udfs() {
        let ctx = SessionContext::new();
        register_rustynum_udfs(&ctx);
        // Verify all 4 UDFs are registered
        assert!(ctx.udf("rusty_hamming").is_ok());
        assert!(ctx.udf("rusty_similarity").is_ok());
        assert!(ctx.udf("rusty_popcount").is_ok());
        assert!(ctx.udf("rusty_xor_bind").is_ok());
    }
}

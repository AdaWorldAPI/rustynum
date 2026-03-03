//! CogRecord ↔ Lance dataset I/O.
//!
//! Write batches of CogRecords to a Lance dataset on disk (or object store),
//! and read them back. Lance stores each container as a `FixedSizeBinary(2048)`
//! column, enabling IVF indexing on any container.
//!
//! All functions are async (Lance uses tokio internally).

use crate::arrow_bridge::{cogrecord_schema, cogrecord_views, cogrecords_to_record_batch};
use arrow::array::RecordBatch;
use arrow::array::RecordBatchIterator;
use futures::StreamExt;
use lance::dataset::write::{WriteMode, WriteParams};
use lance::dataset::Dataset;
use rustynum_rs::CogRecord;
use std::sync::Arc;

/// Write CogRecords to a new Lance dataset at `uri`.
///
/// Creates the dataset if it doesn't exist. Overwrites if it does.
pub async fn write_cogrecords(uri: &str, records: &[CogRecord]) -> Result<Dataset, lance::Error> {
    let batch =
        cogrecords_to_record_batch(records).expect("CogRecord data must be 2048 bytes per channel");
    let schema = Arc::new(cogrecord_schema());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut params = WriteParams::default();
    params.mode = WriteMode::Create;
    Dataset::write(reader, uri, Some(params)).await
}

/// Append CogRecords to an existing Lance dataset.
pub async fn append_cogrecords(uri: &str, records: &[CogRecord]) -> Result<Dataset, lance::Error> {
    let batch =
        cogrecords_to_record_batch(records).expect("CogRecord data must be 2048 bytes per channel");
    let schema = Arc::new(cogrecord_schema());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut params = WriteParams::default();
    params.mode = WriteMode::Append;
    Dataset::write(reader, uri, Some(params)).await
}

/// Read all CogRecords from a Lance dataset.
///
/// This allocates owned `CogRecord`s (16384 bytes each). For zero-copy access,
/// use [`read_cogrecord_batches()`] and iterate with [`cogrecord_views()`].
pub async fn read_cogrecords(uri: &str) -> Result<Vec<CogRecord>, lance::Error> {
    let batches = read_cogrecord_batches(uri).await?;
    let records = batches
        .iter()
        .flat_map(|batch| cogrecord_views(batch).into_iter().map(|v| v.to_owned()))
        .collect();
    Ok(records)
}

/// Read raw Arrow RecordBatches from a Lance dataset (zero-copy friendly).
///
/// Returns the batches as-is. Use [`cogrecord_views()`] to get zero-copy
/// `CogRecordView` references into each batch without any allocation.
///
/// ## Example
///
/// ```rust,ignore
/// let batches = read_cogrecord_batches("data.lance").await?;
/// for batch in &batches {
///     for view in cogrecord_views(batch) {
///         // view.meta, view.cam, etc. borrow directly from Arrow buffers
///     }
/// }
/// ```
pub async fn read_cogrecord_batches(uri: &str) -> Result<Vec<RecordBatch>, lance::Error> {
    let dataset = Dataset::open(uri).await?;
    let mut batches = Vec::new();

    let mut stream = dataset.scan().try_into_stream().await?;

    while let Some(batch_result) = stream.next().await {
        batches.push(batch_result?);
    }

    Ok(batches)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustynum_rs::NumArrayU8;

    fn make_test_records(n: usize) -> Vec<CogRecord> {
        (0..n)
            .map(|i| {
                let val = (i % 256) as u8;
                CogRecord::new(
                    NumArrayU8::new(vec![val; 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(1); 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(2); 2048]),
                    NumArrayU8::new(vec![val.wrapping_add(3); 2048]),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_write_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("test.lance");
        let uri_str = uri.to_str().unwrap();

        let records = make_test_records(100);
        write_cogrecords(uri_str, &records).await.unwrap();

        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 100);

        for (i, rec) in back.iter().enumerate() {
            let val = (i % 256) as u8;
            assert_eq!(rec.meta.data_slice()[0], val);
            assert_eq!(rec.cam.data_slice()[0], val.wrapping_add(1));
            assert_eq!(rec.btree.data_slice()[0], val.wrapping_add(2));
            assert_eq!(rec.embed.data_slice()[0], val.wrapping_add(3));
        }
    }

    #[tokio::test]
    async fn test_append() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("append.lance");
        let uri_str = uri.to_str().unwrap();

        write_cogrecords(uri_str, &make_test_records(50))
            .await
            .unwrap();
        append_cogrecords(uri_str, &make_test_records(50))
            .await
            .unwrap();

        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 100);
    }

    #[tokio::test]
    async fn test_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("empty.lance");
        let uri_str = uri.to_str().unwrap();

        write_cogrecords(uri_str, &[]).await.unwrap();
        let back = read_cogrecords(uri_str).await.unwrap();
        assert_eq!(back.len(), 0);
    }

    #[tokio::test]
    async fn test_read_batches_zero_copy() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("batches.lance");
        let uri_str = uri.to_str().unwrap();

        let records = make_test_records(100);
        write_cogrecords(uri_str, &records).await.unwrap();

        let batches = read_cogrecord_batches(uri_str).await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 100);

        // Zero-copy views borrow directly from Arrow buffers
        let first_batch = &batches[0];
        let views = crate::arrow_bridge::cogrecord_views(first_batch);
        assert!(!views.is_empty());
        assert_eq!(views[0].meta.len(), 2048);
    }
}

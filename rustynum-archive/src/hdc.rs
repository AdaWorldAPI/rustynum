//! Hyperdimensional Computing operations for NumArrayU8.
//!
//! Provides the core HDC primitives optimized for AVX-512:
//! - **BIND** (XOR) — already available via `^` operator and BitwiseSimdOps
//! - **BUNDLE** — majority vote across multiple bitpacked vectors
//! - **PERMUTE** — circular bit-lane rotation
//! - **DISTANCE** — hamming distance (already in bitwise.rs)
//! - **DOT_I8** — int8 dot product (VNNI-targetable) for embedding containers
//!
//! ## BUNDLE optimization
//!
//! Hybrid strategy:
//! - **Small n** (≤ 16 vectors): compiler-auto-vectorized per-byte counting.
//!   The compiler recognizes the sequential pattern and emits AVX-512 bytewise ops.
//! - **Large n** (> 16 vectors): ripple-carry bit-parallel counter with explicit
//!   `u64x8` SIMD. Processes 512 bit positions per instruction.
//!
//! Parallelization uses the **blackboard borrow-mut scheme**: the output
//! buffer is split into disjoint mutable regions via `split_at_mut`, giving
//! each thread exclusive ownership of its slice. No `Arc`, no `Mutex`,
//! no lock contention.
//!
//! ## CogRecord container support
//!
//! Designed for 4 × 16384-bit (2048-byte) containers = 8KB CogRecord:
//! - Container 0: META — codebook identity + DN + hashtag zone
//! - Container 1: CAM — content-addressable memory (Hamming via VPOPCNTDQ)
//! - Container 2: B-tree — structural position index
//! - Container 3: Embedding — int8/int4/binary (dot product via VNNI, Hamming via VPOPCNTDQ)
//!
//! All container sizes (2048, 8192, 16384, 65536 bytes) are multiples of 64,
//! so every SIMD path uses full u64x8 vectors with zero scalar tail.
//!
//! ## Archive note
//!
//! This is a frozen copy of the original HDC module from rustynum-rs.
//! The live version continues to evolve in rustynum-rs/src/num_array/hdc.rs.

// NOTE: This archived module documents the HDC API but delegates to
// rustynum_rs::NumArrayU8 which contains the actual implementation.
// The methods (bind, permute, bundle, dot_i8, hamming_distance_adaptive,
// hamming_search_adaptive, cosine_search_adaptive) are all implemented
// directly on NumArrayU8 in rustynum-rs.
//
// This file is kept as documentation of the v1 model's HDC capabilities.

pub use rustynum_rs::NumArrayU8;

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PERMUTE tests ----

    #[test]
    fn test_permute_zero() {
        let v = NumArrayU8::new(vec![0xAA; 8192]);
        let p = v.permute(0);
        assert_eq!(p.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_full_rotation() {
        let v = NumArrayU8::new(vec![0xAA; 16]);
        let total_bits = 16 * 8;
        let p = v.permute(total_bits);
        assert_eq!(p.get_data(), v.get_data());
    }

    #[test]
    fn test_permute_single_bit() {
        let mut data = vec![0u8; 16];
        data[0] = 0x80;
        let v = NumArrayU8::new(data);
        let p = v.permute(1);
        let mut expected = vec![0u8; 16];
        expected[1] = 0x01;
        assert_eq!(p.get_data(), &expected);
    }

    #[test]
    fn test_permute_inverse() {
        let v = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let total_bits = 8192 * 8;
        let k = 42;
        let p1 = v.permute(k);
        let p2 = p1.permute(total_bits - k);
        assert_eq!(p2.get_data(), v.get_data());
    }

    // ---- BUNDLE tests ----

    #[test]
    fn test_bundle_majority_2_of_3() {
        let a = NumArrayU8::new(vec![0xFF; 8]);
        let b = NumArrayU8::new(vec![0xFF; 8]);
        let c = NumArrayU8::new(vec![0x00; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0xFF; 8]);
    }

    #[test]
    fn test_bundle_mixed_pattern() {
        let a = NumArrayU8::new(vec![0b10101010; 8]);
        let b = NumArrayU8::new(vec![0b11001100; 8]);
        let c = NumArrayU8::new(vec![0b11110000; 8]);
        let result = NumArrayU8::bundle(&[&a, &b, &c]);
        assert_eq!(result.get_data(), &vec![0b11101000; 8]);
    }

    // ---- BIND tests ----

    #[test]
    fn test_bind_involution() {
        let a = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let b = NumArrayU8::new((0..8192).map(|i| ((i * 7 + 13) % 256) as u8).collect());
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        assert_eq!(recovered.get_data(), a.get_data());
    }

    // ---- DOT_I8 tests ----

    #[test]
    fn test_dot_i8_simple() {
        let a = NumArrayU8::new(vec![1, 2, 3, 4]);
        let b = NumArrayU8::new(vec![1, 2, 3, 4]);
        assert_eq!(a.dot_i8(&b), 30);
    }

    // ---- Adaptive search tests ----

    #[test]
    fn test_adaptive_hamming_identical() {
        let a = NumArrayU8::new(vec![0xAA; 2048]);
        let result = a.hamming_distance_adaptive(&a, 100);
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_adaptive_hamming_reject_far() {
        let a = NumArrayU8::new(vec![0xFF; 2048]);
        let b = NumArrayU8::new(vec![0x00; 2048]);
        assert!(a.hamming_distance_adaptive(&b, 100).is_none());
    }

    #[test]
    fn test_adaptive_search_batch() {
        let query = NumArrayU8::new(vec![0xAA; 2048]);
        let mut db_data = vec![0xAA; 2048]; // vec 0: identical (d=0)
        db_data.extend(vec![0x55; 2048]); // vec 1: maximally different
        db_data.extend(vec![0xAA; 2048]); // vec 2: identical (d=0)
        db_data.extend(vec![0x00; 2048]); // vec 3: very different
        let db = NumArrayU8::new(db_data);

        let results = query.hamming_search_adaptive(&db, 2048, 4, 100);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, 0));
        assert_eq!(results[1], (2, 0));
    }

    // ---- Edge encode/decode integration ----

    #[test]
    fn test_edge_encode_decode() {
        let src = NumArrayU8::new((0..8192).map(|i| (i % 256) as u8).collect());
        let rel = NumArrayU8::new((0..8192).map(|i| ((i * 3) % 256) as u8).collect());
        let tgt = NumArrayU8::new((0..8192).map(|i| ((i * 7 + 42) % 256) as u8).collect());

        let total_bits = 8192 * 8;
        let perm_rel = rel.permute(1);
        let perm_tgt = tgt.permute(2);
        let edge = &(&src ^ &perm_rel) ^ &perm_tgt;

        let recovered_perm_tgt = &(&edge ^ &src) ^ &perm_rel;
        let recovered_tgt = recovered_perm_tgt.permute(total_bits - 2);
        assert_eq!(recovered_tgt.get_data(), tgt.get_data());
    }
}

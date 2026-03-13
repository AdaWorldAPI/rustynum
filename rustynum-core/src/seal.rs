//! Seal: integrity verification via blake3. Integer comparison only.
//!
//! [`MerkleRoot`] is a 48-bit truncated blake3 hash of data masked by alpha.
//! Undefined bits (alpha=0) are zeroed before hashing, so noise in undefined
//! regions doesn't change the hash.
//!
//! [`Seal`] is the verification result: `Wisdom` (matches) or `Staunen` (differs).
//! No float. No NaN. Byte equality only.

use crate::plane::Plane;

// ============================================================================
// Seal — the verification result
// ============================================================================

/// Integrity seal. Byte equality comparison of blake3 hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seal {
    /// Hash matches stored hash. Stable. Consolidated.
    Wisdom,
    /// Hash differs from stored hash. Changed. Surprising.
    Staunen,
}

// ============================================================================
// MerkleRoot — truncated blake3 hash
// ============================================================================

/// Truncated blake3 hash for compact storage.
/// 48 bits = collision-safe for <10M nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MerkleRoot(pub [u8; 6]);

impl MerkleRoot {
    /// Create from raw bytes.
    #[inline]
    pub fn from_bytes(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }

    /// View as byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

impl Plane {
    /// Compute merkle root from data masked by alpha.
    /// Undefined bits (alpha=0) are zeroed before hashing.
    /// This means noise in undefined regions doesn't change the hash.
    pub fn merkle(&mut self) -> MerkleRoot {
        self.ensure_cache();
        let mut masked = vec![0u8; Self::BYTES];
        let bits_bytes = self.bits_bytes_ref();
        let alpha_bytes = self.alpha_bytes_ref();
        for i in 0..Self::BYTES {
            masked[i] = bits_bytes[i] & alpha_bytes[i];
        }
        let hash = blake3::hash(&masked);
        let mut root = [0u8; 6];
        root.copy_from_slice(&hash.as_bytes()[..6]);
        MerkleRoot(root)
    }

    /// Verify integrity against a stored root.
    pub fn verify(&mut self, stored: &MerkleRoot) -> Seal {
        if self.merkle() == *stored {
            Seal::Wisdom
        } else {
            Seal::Staunen
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_deterministic() {
        let mut p = Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        let root1 = p.merkle();
        let root2 = p.merkle();
        assert_eq!(root1, root2);
    }

    #[test]
    fn seal_stable_despite_recompute() {
        let mut p = Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        let root = p.merkle();
        assert_eq!(p.verify(&root), Seal::Wisdom);
    }

    #[test]
    fn seal_detects_change() {
        let mut p = Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        let root = p.merkle();

        // New encounter changes the plane
        p.encounter("world");
        assert_eq!(p.verify(&root), Seal::Staunen);
    }

    #[test]
    fn empty_plane_merkle() {
        let mut p = Plane::new();
        let root = p.merkle();
        // Empty plane → all zeros masked → deterministic hash of zeros
        assert_eq!(p.verify(&root), Seal::Wisdom);
    }
}

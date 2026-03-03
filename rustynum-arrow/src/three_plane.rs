//! Three-plane dual-layer buffer types for the bind_nodes_v2 schema.
//!
//! Provides zero-copy buffer wrappers for the three-plane (S/P/O)
//! dual-layer (binary + soaking) architecture. Each plane has:
//! - A binary structural layer: `Fingerprint<256>` = 2048 bytes
//! - An optional int8 soaking layer: 10000 dimensions (nullable when crystallized)
//!
//! # Schema
//!
//! The `bind_nodes_v2_schema()` function defines the Arrow schema for the
//! three-plane node table, compatible with Lance dataset storage.
//!
//! # Zero-Copy
//!
//! `SoakingBuffer` wraps a contiguous `Vec<i8>` for owned data or
//! provides views into Arrow/Lance mmap'd buffers via `SoakingView`.
//! No copies between representations within a single binary.

use rustynum_core::fingerprint::Fingerprint;
use rustynum_core::organic::SynapseState;
use rustynum_core::soaking::{self, SOAKING_DIM};

#[cfg(feature = "arrow")]
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
#[cfg(feature = "arrow")]
use std::sync::Arc;

/// Binary structural layer size: 16384 bits = 256 × u64 = 2048 bytes.
pub const BINARY_BYTES: usize = 2048;

/// Default soaking dimension: 10000 int8 values per plane.
pub const DEFAULT_SOAKING_DIM: usize = SOAKING_DIM;

// ---------------------------------------------------------------------------
// Schema definitions
// ---------------------------------------------------------------------------

/// Arrow schema for bind_nodes_v2: three-plane dual-layer node table.
///
/// Each node has three planes (S/P/O) with binary + optional soaking layers,
/// sigma bands, gate states, NARS evidence, and metadata.
#[cfg(feature = "arrow")]
pub fn bind_nodes_v2_schema() -> ArrowSchema {
    let soaking_dim = DEFAULT_SOAKING_DIM as i32;
    let binary_bytes = BINARY_BYTES as i32;

    ArrowSchema::new(vec![
        // Addressing
        Field::new("addr", DataType::UInt16, false),
        Field::new("label", DataType::Utf8, true),
        // S Plane (Subject)
        Field::new(
            "s_binary",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new(
            "s_soaking",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int8, false)),
                soaking_dim,
            ),
            true, // nullable: only present during active soaking
        ),
        // P Plane (Predicate)
        Field::new(
            "p_binary",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new(
            "p_soaking",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int8, false)),
                soaking_dim,
            ),
            true,
        ),
        // O Plane (Object)
        Field::new(
            "o_binary",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new(
            "o_soaking",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int8, false)),
                soaking_dim,
            ),
            true,
        ),
        // Composite (derived, for backward compat + CAM)
        Field::new(
            "spo_binary",
            DataType::FixedSizeBinary(binary_bytes),
            true,
        ),
        // Sigma mask (per plane)
        Field::new("s_sigma", DataType::UInt8, false),
        Field::new("p_sigma", DataType::UInt8, false),
        Field::new("o_sigma", DataType::UInt8, false),
        // NARS evidence
        Field::new("nars_f", DataType::Float32, false),
        Field::new("nars_c", DataType::Float32, false),
        Field::new("evidence_count", DataType::UInt32, false),
        // Gate state per plane (0=BLOCK, 1=HOLD, 2=FLOW)
        Field::new("s_gate", DataType::UInt8, false),
        Field::new("p_gate", DataType::UInt8, false),
        Field::new("o_gate", DataType::UInt8, false),
        // Role assignment provenance (bitflags)
        Field::new("role_provenance", DataType::UInt8, false),
        // Metadata
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

/// Arrow schema for the global attention mask table.
#[cfg(feature = "arrow")]
pub fn attention_mask_schema() -> ArrowSchema {
    let soaking_dim = DEFAULT_SOAKING_DIM as i32;

    ArrowSchema::new(vec![
        Field::new(
            "mask",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int8, false)),
                soaking_dim,
            ),
            false,
        ),
        Field::new("concept_count", DataType::UInt32, false),
        Field::new("min_sigma", DataType::UInt8, false),
        Field::new("rebuilt_at", DataType::UInt64, false),
        Field::new("version", DataType::UInt64, false),
    ])
}

/// Arrow schema for bind_edges_v2: per-plane edge fingerprints.
#[cfg(feature = "arrow")]
pub fn bind_edges_v2_schema() -> ArrowSchema {
    let binary_bytes = BINARY_BYTES as i32;

    ArrowSchema::new(vec![
        Field::new("from_addr", DataType::UInt16, false),
        Field::new("to_addr", DataType::UInt16, false),
        Field::new("verb_addr", DataType::UInt16, false),
        // Per-plane edge fingerprints
        Field::new(
            "edge_s",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new(
            "edge_p",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new(
            "edge_o",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        // Composite (backward compat)
        Field::new(
            "fingerprint",
            DataType::FixedSizeBinary(binary_bytes),
            false,
        ),
        Field::new("weight", DataType::Float32, false),
        Field::new("nars_f", DataType::Float32, false),
        Field::new("nars_c", DataType::Float32, false),
    ])
}

// ---------------------------------------------------------------------------
// Role provenance bitflags
// ---------------------------------------------------------------------------

/// How S/P/O role was decided. Stored as bitflags in `role_provenance`.
pub mod role_provenance {
    /// Role assigned by grammar position (SVO word order).
    pub const GRAMMAR: u8 = 0x01;
    /// Role assigned by NSM prime decomposition.
    pub const NSM: u8 = 0x02;
    /// Role refined by σ-2/3 background knowledge context.
    pub const SIGMA_CONTEXT: u8 = 0x04;
    /// Role explicitly assigned by user.
    pub const USER_EXPLICIT: u8 = 0x08;
}

/// Gate state values for s_gate/p_gate/o_gate columns.
pub mod gate_state {
    /// Superposition discarded, ground unchanged.
    pub const BLOCK: u8 = 0;
    /// Superposition persists, accumulate more evidence.
    pub const HOLD: u8 = 1;
    /// Superposition collapsed to ground truth.
    pub const FLOW: u8 = 2;
}

// ---------------------------------------------------------------------------
// SoakingBuffer
// ---------------------------------------------------------------------------

/// Owned int8 soaking buffer: contiguous storage for N entries × D dimensions.
///
/// Each entry is a `dim`-length int8 vector representing the soaking register
/// for one concept in one plane (S, P, or O).
pub struct SoakingBuffer {
    data: Vec<i8>,
    dim: usize,
    len: usize,
}

impl SoakingBuffer {
    /// Create a new zero-initialized soaking buffer.
    pub fn new(len: usize, dim: usize) -> Self {
        Self {
            data: vec![0i8; len * dim],
            dim,
            len,
        }
    }

    /// Create from an existing data vector.
    ///
    /// Panics if `data.len() != len * dim`.
    pub fn from_data(data: Vec<i8>, dim: usize, len: usize) -> Self {
        assert_eq!(data.len(), len * dim, "data length must be len × dim");
        Self { data, dim, len }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Soaking dimension per entry.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get soaking register at index (zero-copy borrow).
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[i8]> {
        if index >= self.len {
            return None;
        }
        let offset = index * self.dim;
        Some(&self.data[offset..offset + self.dim])
    }

    /// Get mutable soaking register at index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [i8]> {
        if index >= self.len {
            return None;
        }
        let offset = index * self.dim;
        Some(&mut self.data[offset..offset + self.dim])
    }

    /// Deposit evidence into a register via saturating add.
    pub fn deposit(&mut self, index: usize, evidence: &[i8], weight: f32) {
        assert!(index < self.len, "index out of bounds");
        assert_eq!(evidence.len(), self.dim, "evidence dim must match buffer dim");
        let offset = index * self.dim;
        for i in 0..self.dim {
            let current = self.data[offset + i];
            let contribution = (evidence[i] as f32 * weight).round() as i16;
            self.data[offset + i] = (current as i16 + contribution).clamp(-128, 127) as i8;
        }
    }

    /// Evaluate saturation ratio for a register.
    ///
    /// Returns fraction of dimensions with |value| > threshold.
    pub fn saturation_ratio(&self, index: usize, threshold: u8) -> f32 {
        if let Some(register) = self.get(index) {
            let saturated = register
                .iter()
                .filter(|&&v| v.unsigned_abs() > threshold)
                .count();
            saturated as f32 / self.dim as f32
        } else {
            0.0
        }
    }

    /// Crystallize register at index: sign(soaking) → binary Fingerprint<256>.
    pub fn crystallize(&self, index: usize) -> Option<Fingerprint<256>> {
        let register = self.get(index)?;
        Some(soaking::int8_to_binary(register))
    }

    /// Get raw data slice.
    #[inline]
    pub fn as_slice(&self) -> &[i8] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// ThreePlaneFingerprintBuffer
// ---------------------------------------------------------------------------

/// A single plane's state: binary structural layer + optional soaking layer.
pub struct PlaneBuffer {
    /// Binary structural fingerprints: len × 2048 bytes each.
    binary: Vec<Fingerprint<256>>,
    /// Optional soaking registers (None = all entries crystallized in this plane).
    soaking: Option<SoakingBuffer>,
}

impl PlaneBuffer {
    /// Create a new plane buffer with all-zero fingerprints and no soaking.
    pub fn new_crystallized(len: usize) -> Self {
        Self {
            binary: vec![Fingerprint::zero(); len],
            soaking: None,
        }
    }

    /// Create a new plane buffer with soaking enabled.
    pub fn new_with_soaking(len: usize, dim: usize) -> Self {
        Self {
            binary: vec![Fingerprint::zero(); len],
            soaking: Some(SoakingBuffer::new(len, dim)),
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.binary.len()
    }

    /// Whether this plane is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.binary.is_empty()
    }

    /// Get binary fingerprint at index.
    #[inline]
    pub fn binary(&self, index: usize) -> &Fingerprint<256> {
        &self.binary[index]
    }

    /// Get mutable binary fingerprint at index.
    #[inline]
    pub fn binary_mut(&mut self, index: usize) -> &mut Fingerprint<256> {
        &mut self.binary[index]
    }

    /// Whether soaking is active for this plane.
    #[inline]
    pub fn has_soaking(&self) -> bool {
        self.soaking.is_some()
    }

    /// Get soaking buffer reference (if active).
    #[inline]
    pub fn soaking(&self) -> Option<&SoakingBuffer> {
        self.soaking.as_ref()
    }

    /// Get mutable soaking buffer (if active).
    #[inline]
    pub fn soaking_mut(&mut self) -> Option<&mut SoakingBuffer> {
        self.soaking.as_mut()
    }

    /// Enable soaking for this plane.
    pub fn enable_soaking(&mut self, dim: usize) {
        if self.soaking.is_none() {
            self.soaking = Some(SoakingBuffer::new(self.len(), dim));
        }
    }

    /// Crystallize a specific entry: sign(soaking) → binary, then clear soaking.
    ///
    /// Returns the crystallized fingerprint, or None if soaking is not active.
    pub fn crystallize_entry(&mut self, index: usize) -> Option<Fingerprint<256>> {
        let fp = self.soaking.as_ref()?.crystallize(index)?;
        self.binary[index] = fp.clone();
        // Zero out the soaking register for this entry
        if let Some(ref mut soaking) = self.soaking {
            if let Some(reg) = soaking.get_mut(index) {
                reg.fill(0);
            }
        }
        Some(fp)
    }
}

/// Three-plane dual-layer buffer: S, P, O planes each with binary + soaking.
///
/// This is the working-set representation for the bind_nodes_v2 schema.
/// Each plane holds binary structural fingerprints and optional int8 soaking.
pub struct ThreePlaneFingerprintBuffer {
    /// Subject plane.
    pub s: PlaneBuffer,
    /// Predicate plane.
    pub p: PlaneBuffer,
    /// Object plane.
    pub o: PlaneBuffer,
    /// Number of entries (uniform across planes).
    len: usize,
}

impl ThreePlaneFingerprintBuffer {
    /// Create a new buffer with all entries crystallized (no soaking).
    pub fn new_crystallized(len: usize) -> Self {
        Self {
            s: PlaneBuffer::new_crystallized(len),
            p: PlaneBuffer::new_crystallized(len),
            o: PlaneBuffer::new_crystallized(len),
            len,
        }
    }

    /// Create a new buffer with soaking enabled on all planes.
    pub fn new_with_soaking(len: usize, dim: usize) -> Self {
        Self {
            s: PlaneBuffer::new_with_soaking(len, dim),
            p: PlaneBuffer::new_with_soaking(len, dim),
            o: PlaneBuffer::new_with_soaking(len, dim),
            len,
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get all three binary fingerprints for an entry.
    pub fn binary_triple(&self, index: usize) -> (&Fingerprint<256>, &Fingerprint<256>, &Fingerprint<256>) {
        (self.s.binary(index), self.p.binary(index), self.o.binary(index))
    }

    /// Compute composite SPO binary via XOR role binding.
    ///
    /// `spo_binary = s_binary ⊗ role_s ⊕ p_binary ⊗ role_p ⊕ o_binary ⊗ role_o`
    pub fn compute_spo_binary(
        &self,
        index: usize,
        role_s: &Fingerprint<256>,
        role_p: &Fingerprint<256>,
        role_o: &Fingerprint<256>,
    ) -> Fingerprint<256> {
        let s_bound = self.s.binary(index) ^ role_s;
        let p_bound = self.p.binary(index) ^ role_p;
        let o_bound = self.o.binary(index) ^ role_o;
        &(&s_bound ^ &p_bound) ^ &o_bound
    }
}

// ---------------------------------------------------------------------------
// Organic soaking buffer (prompt 19 integration)
// ---------------------------------------------------------------------------

/// Organic soaking buffer: SynapseState per dimension instead of bare i8.
///
/// This is the BCM-enhanced version of SoakingBuffer, using the organic
/// plasticity model from prompt 19. Each dimension has efficacy, theta,
/// and maturity instead of a single i8 value.
pub struct OrganicSoakingBuffer {
    data: Vec<SynapseState>,
    dim: usize,
    len: usize,
}

impl OrganicSoakingBuffer {
    /// Create a new organic soaking buffer with fresh synapses.
    pub fn new(len: usize, dim: usize) -> Self {
        Self {
            data: vec![SynapseState::new(); len * dim],
            dim,
            len,
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Soaking dimension per entry.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get SynapseState register at index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[SynapseState]> {
        if index >= self.len {
            return None;
        }
        let offset = index * self.dim;
        Some(&self.data[offset..offset + self.dim])
    }

    /// Get mutable SynapseState register at index.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut [SynapseState]> {
        if index >= self.len {
            return None;
        }
        let offset = index * self.dim;
        Some(&mut self.data[offset..offset + self.dim])
    }

    /// Organic deposit: BCM-style plasticity update for a register.
    pub fn organic_deposit(&mut self, index: usize, evidence: &[i8]) {
        assert!(index < self.len, "index out of bounds");
        assert_eq!(evidence.len(), self.dim, "evidence dim must match buffer dim");
        let offset = index * self.dim;
        let register = &mut self.data[offset..offset + self.dim];
        rustynum_core::organic::organic_deposit_batch(register, evidence);
    }

    /// Run homeostatic scaling across a register.
    pub fn homeostatic_scale(&mut self, index: usize, target_mean: u8, scale_rate: u8) {
        if let Some(register) = self.get_mut(index) {
            rustynum_core::organic::homeostatic_scale(register, target_mean, scale_rate);
        }
    }

    /// Evaluate saturation ratio for a register.
    pub fn saturation_ratio(&self, index: usize) -> f32 {
        if let Some(register) = self.get(index) {
            rustynum_core::organic::saturation_ratio(register, 100)
        } else {
            0.0
        }
    }

    /// Crystallize register: SynapseState → binary Fingerprint<256>.
    pub fn crystallize(&self, index: usize) -> Option<Fingerprint<256>> {
        let register = self.get(index)?;
        Some(rustynum_core::organic::crystallize(register))
    }

    /// Crystallize using 5-state quantization.
    pub fn crystallize_quantized(&self, index: usize) -> Option<Fingerprint<256>> {
        let register = self.get(index)?;
        Some(rustynum_core::organic::crystallize_quantized(register))
    }

    /// Get 5-state histogram for a register (diagnostic).
    pub fn five_state_histogram(&self, index: usize) -> Option<[usize; 5]> {
        let register = self.get(index)?;
        Some(rustynum_core::organic::five_state_histogram(register))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soaking_buffer_basic() {
        let buf = SoakingBuffer::new(10, 100);
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.dim(), 100);
        assert!(!buf.is_empty());
        let reg = buf.get(0).unwrap();
        assert_eq!(reg.len(), 100);
        assert!(reg.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_soaking_buffer_deposit() {
        let mut buf = SoakingBuffer::new(2, 4);
        let evidence = vec![10i8, -5, 3, -1];
        buf.deposit(0, &evidence, 1.0);
        let reg = buf.get(0).unwrap();
        assert_eq!(reg[0], 10);
        assert_eq!(reg[1], -5);
        assert_eq!(reg[2], 3);
        assert_eq!(reg[3], -1);
        // Second entry unchanged
        let reg2 = buf.get(1).unwrap();
        assert!(reg2.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_soaking_buffer_crystallize() {
        let mut buf = SoakingBuffer::new(1, 8);
        let evidence = vec![10i8, -5, 3, -1, 0, 7, -3, 1];
        buf.deposit(0, &evidence, 1.0);
        let fp = buf.crystallize(0).unwrap();
        // Positive → 1, zero/negative → 0
        // bits: 1,0,1,0,0,1,0,1 = 0b10100101 = 0xA5
        assert_eq!(fp.words[0] & 0xFF, 0b10100101);
    }

    #[test]
    fn test_soaking_buffer_saturation() {
        let mut buf = SoakingBuffer::new(1, 4);
        let evidence = vec![120i8, -110, 5, -3];
        buf.deposit(0, &evidence, 1.0);
        let ratio = buf.saturation_ratio(0, 100);
        assert!((ratio - 0.5).abs() < f32::EPSILON); // 2 of 4 above threshold
    }

    #[test]
    fn test_plane_buffer_crystallized() {
        let plane = PlaneBuffer::new_crystallized(5);
        assert_eq!(plane.len(), 5);
        assert!(!plane.has_soaking());
        assert!(plane.binary(0).is_zero());
    }

    #[test]
    fn test_plane_buffer_with_soaking() {
        let mut plane = PlaneBuffer::new_with_soaking(5, 100);
        assert!(plane.has_soaking());
        assert!(plane.soaking().unwrap().get(0).unwrap().iter().all(|&v| v == 0));

        // Deposit and crystallize
        let evidence = vec![50i8; 100];
        plane.soaking_mut().unwrap().deposit(0, &evidence, 1.0);
        let fp = plane.crystallize_entry(0).unwrap();
        // All positive → all first 100 bits should be 1
        for i in 0..100 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            assert_eq!((fp.words[word_idx] >> bit_idx) & 1, 1, "bit {i} should be 1");
        }
        // Binary updated
        assert_eq!(*plane.binary(0), fp);
    }

    #[test]
    fn test_three_plane_buffer_new() {
        let buf = ThreePlaneFingerprintBuffer::new_crystallized(10);
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_empty());
        let (s, p, o) = buf.binary_triple(0);
        assert!(s.is_zero());
        assert!(p.is_zero());
        assert!(o.is_zero());
    }

    #[test]
    fn test_three_plane_buffer_with_soaking() {
        let buf = ThreePlaneFingerprintBuffer::new_with_soaking(5, 100);
        assert!(buf.s.has_soaking());
        assert!(buf.p.has_soaking());
        assert!(buf.o.has_soaking());
    }

    #[test]
    fn test_three_plane_spo_composite() {
        let mut buf = ThreePlaneFingerprintBuffer::new_crystallized(1);
        // Set distinct fingerprints
        *buf.s.binary_mut(0) = Fingerprint::from_words({
            let mut w = [0u64; 256];
            w[0] = 0xFF;
            w
        });
        *buf.p.binary_mut(0) = Fingerprint::from_words({
            let mut w = [0u64; 256];
            w[0] = 0xF0;
            w
        });
        *buf.o.binary_mut(0) = Fingerprint::from_words({
            let mut w = [0u64; 256];
            w[0] = 0x0F;
            w
        });

        // Role vectors
        let role_s = Fingerprint::zero();
        let role_p = Fingerprint::zero();
        let role_o = Fingerprint::zero();

        let spo = buf.compute_spo_binary(0, &role_s, &role_p, &role_o);
        // With zero role vectors, spo = s ^ p ^ o
        let expected = 0xFF ^ 0xF0 ^ 0x0F;
        assert_eq!(spo.words[0], expected);
    }

    #[test]
    fn test_organic_soaking_buffer() {
        let mut buf = OrganicSoakingBuffer::new(2, 10);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.dim(), 10);

        // Deposit evidence
        let evidence = vec![10i8, -5, 3, -1, 0, 7, -3, 1, 8, -2];
        buf.organic_deposit(0, &evidence);

        let reg = buf.get(0).unwrap();
        assert!(reg[0].efficacy > 0); // positive evidence → positive efficacy
        assert!(reg[1].efficacy < 0); // negative evidence → negative efficacy
        assert_eq!(reg[4].efficacy, 0); // zero evidence → no change

        // Second entry unchanged
        let reg2 = buf.get(1).unwrap();
        assert!(reg2.iter().all(|s| s.efficacy == 0));
    }

    #[test]
    fn test_organic_soaking_crystallize() {
        let mut buf = OrganicSoakingBuffer::new(1, 8);
        let evidence = vec![50i8, -50, 30, -30, 0, 40, -20, 10];
        buf.organic_deposit(0, &evidence);

        let fp = buf.crystallize(0).unwrap();
        // Positive efficacy → bit 1, negative/zero → bit 0
        let reg = buf.get(0).unwrap();
        for i in 0..8 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let bit = (fp.words[word_idx] >> bit_idx) & 1;
            if reg[i].efficacy > 0 {
                assert_eq!(bit, 1, "dim {i} has positive efficacy, should be 1");
            } else {
                assert_eq!(bit, 0, "dim {i} has non-positive efficacy, should be 0");
            }
        }
    }

    #[test]
    fn test_organic_soaking_homeostatic() {
        let mut buf = OrganicSoakingBuffer::new(1, 10);
        // Deposit strong evidence repeatedly
        let evidence = vec![50i8; 10];
        for _ in 0..20 {
            buf.organic_deposit(0, &evidence);
        }
        let theta_sum_before: u32 = buf.get(0).unwrap().iter().map(|s| s.theta as u32).sum();

        // Homeostatic scale should adjust theta
        buf.homeostatic_scale(0, 30, 2);
        let theta_sum_after: u32 = buf.get(0).unwrap().iter().map(|s| s.theta as u32).sum();

        // Mean |efficacy| is high → theta should have increased
        let mean_eff = rustynum_core::organic::mean_efficacy(buf.get(0).unwrap());
        if mean_eff > 30.0 {
            assert!(
                theta_sum_after >= theta_sum_before,
                "theta should increase when mean efficacy is above target"
            );
        }
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_bind_nodes_v2_schema() {
        let schema = bind_nodes_v2_schema();
        assert!(schema.field_with_name("s_binary").is_ok());
        assert!(schema.field_with_name("p_binary").is_ok());
        assert!(schema.field_with_name("o_binary").is_ok());
        assert!(schema.field_with_name("s_soaking").is_ok());
        assert!(schema.field_with_name("spo_binary").is_ok());
        assert!(schema.field_with_name("s_sigma").is_ok());
        assert!(schema.field_with_name("s_gate").is_ok());
        assert!(schema.field_with_name("role_provenance").is_ok());
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_attention_mask_schema() {
        let schema = attention_mask_schema();
        assert!(schema.field_with_name("mask").is_ok());
        assert!(schema.field_with_name("concept_count").is_ok());
        assert!(schema.field_with_name("version").is_ok());
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_bind_edges_v2_schema() {
        let schema = bind_edges_v2_schema();
        assert!(schema.field_with_name("edge_s").is_ok());
        assert!(schema.field_with_name("edge_p").is_ok());
        assert!(schema.field_with_name("edge_o").is_ok());
        assert!(schema.field_with_name("fingerprint").is_ok());
    }

    #[test]
    fn test_role_provenance_flags() {
        let flags = role_provenance::GRAMMAR | role_provenance::NSM | role_provenance::SIGMA_CONTEXT;
        assert_eq!(flags, 0x07);
        assert!(flags & role_provenance::GRAMMAR != 0);
        assert!(flags & role_provenance::USER_EXPLICIT == 0);
    }

    #[test]
    fn test_gate_state_values() {
        assert_eq!(gate_state::BLOCK, 0);
        assert_eq!(gate_state::HOLD, 1);
        assert_eq!(gate_state::FLOW, 2);
    }

    #[test]
    fn test_soaking_buffer_out_of_bounds() {
        let buf = SoakingBuffer::new(2, 4);
        assert!(buf.get(2).is_none());
        assert!(buf.get(100).is_none());
    }

    #[test]
    fn test_organic_five_state_histogram() {
        let mut buf = OrganicSoakingBuffer::new(1, 5);
        // Deposit diverse evidence
        let evidence = vec![80i8, -80, 0, 30, -30];
        for _ in 0..20 {
            buf.organic_deposit(0, &evidence);
        }
        let hist = buf.five_state_histogram(0).unwrap();
        // Should have non-zero counts across at least 2 states
        let non_zero = hist.iter().filter(|&&c| c > 0).count();
        assert!(non_zero >= 2, "histogram should have diverse states: {:?}", hist);
    }
}

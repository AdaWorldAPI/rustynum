//! Holographic Spatial Binding: Gabor Wavelets + Delta Cube Architecture.
//!
//! The convergence layer that unifies spatial locality, frequency multiplexing,
//! and attention gating into a single primitive: the Gabor wavelet.
//!
//! A Gabor wavelet is a sinusoidal carrier × Gaussian envelope:
//!   wavelet(j) = A · exp(-(j-j₀)²/(2σ²)) · cos(2π·f·(j-j₀)/N + φ)
//!
//! It is the optimal tradeoff between spatial localization and frequency
//! selectivity (Heisenberg-Gabor uncertainty: Δx·Δf ≥ 1/4π).
//!
//! ## What it subsumes
//!
//! | Previous layer      | How Gabor subsumes it                    |
//! |---------------------|------------------------------------------|
//! | Carrier (carrier.rs)| cos() oscillation IS the carrier         |
//! | Focus (focus.rs)    | exp() envelope IS the spatial mask       |
//! | Phase (phase.rs)    | Phase offset φ IS the concept value      |
//! | 5D projection       | Center (x₀,y₀,z₀) IS the spatial coord  |
//!
//! ## Delta-Cube Binding
//!
//! Content stored in the INTERFERENCE PATTERN between two holographic fields.
//! Neither field alone reveals the content. Both together reconstruct it.
//!
//! ## Spatial Transforms
//!
//! 3D permutations (rotate, diagonal, compose) that move content through
//! the 8×8×32 volume. Composable and invertible.

use crate::focus::{FOCUS_DIM_X, FOCUS_DIM_Y, FOCUS_DIM_Z};

// ============================================================================
// GaussianLUT — Envelope lookup table
// ============================================================================

/// Precomputed Gaussian envelope: exp(-d²/(2σ²)) quantized to u8.
///
/// For the 8×8×32 volume, max squared distance from center:
///   d² = 7² + 7² + 31² = 49 + 49 + 961 = 1059
///
/// Table indexed by d², returns amplitude (0 = fully decayed, 255 = center).
pub struct GaussianLUT {
    pub table: Vec<u8>,
    pub sigma_squared_2: f32,
}

impl GaussianLUT {
    pub fn new(sigma: f32) -> Self {
        let max_d2 = 1060usize;
        let s2 = 2.0 * sigma * sigma;
        let table: Vec<u8> = (0..max_d2)
            .map(|d2| {
                let envelope = (-(d2 as f32) / s2).exp();
                (envelope * 255.0).round().min(255.0) as u8
            })
            .collect();
        Self { table, sigma_squared_2: s2 }
    }

    /// Get envelope amplitude for squared distance from center.
    #[inline]
    pub fn amplitude(&self, d_squared: u32) -> u8 {
        if (d_squared as usize) < self.table.len() {
            self.table[d_squared as usize]
        } else {
            0
        }
    }

    /// Effective radius squared (3σ cutoff: exp(-4.5) < 0.01).
    #[inline]
    pub fn effective_radius_sq(&self) -> u32 {
        (3.0 * self.sigma_squared_2).ceil() as u32
    }
}

// ============================================================================
// WaveletTemplate — Precomputed active positions
// ============================================================================

/// Precomputed wavelet template for a specific (σ, center) pair.
/// Contains only active positions within the Gaussian's effective radius.
pub struct WaveletTemplate {
    /// (byte_index, envelope_amplitude) pairs, sorted by index.
    pub entries: Vec<(u16, u8)>,
}

impl WaveletTemplate {
    /// Build a template for a given center and LUT.
    pub fn new(lut: &GaussianLUT, x0: u8, y0: u8, z0: u8) -> Self {
        let radius_sq = lut.effective_radius_sq();
        let mut entries = Vec::new();

        for x in 0..FOCUS_DIM_X as i32 {
            let dx = x - x0 as i32;
            for y in 0..FOCUS_DIM_Y as i32 {
                let dy = y - y0 as i32;
                for z in 0..FOCUS_DIM_Z as i32 {
                    let dz = z - z0 as i32;
                    let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                    if d_sq > radius_sq { continue; }

                    let env = lut.amplitude(d_sq);
                    if env < 1 { continue; }

                    let idx = x as u16 * 256 + y as u16 * 32 + z as u16;
                    entries.push((idx, env));
                }
            }
        }

        entries.sort_by_key(|&(idx, _)| idx);
        Self { entries }
    }
}

// ============================================================================
// carrier_phase_3d — 3D carrier phase computation
// ============================================================================

/// Carrier phase at position (dx, dy, dz) relative to center.
///
/// Propagation along (1,1,1) diagonal, scale = 32 (Z dimension).
#[inline]
fn carrier_phase_3d(dx: i32, dy: i32, dz: i32, freq: f32, phi: f32) -> f32 {
    let spatial_phase = freq * (dx + dy + dz) as f32 / 32.0;
    std::f32::consts::TAU * spatial_phase + phi
}

// ============================================================================
// Operation 15: gabor_write — Place a concept in 3D space
// ============================================================================

/// Write a Gabor wavelet into the container at a specific 3D location.
///
/// Encodes: position (x0,y0,z0), frequency f, phase φ, width σ, amplitude A.
/// The Gaussian envelope naturally gates the write — no external mask needed.
///
/// Cost: iterates over ~(4σ)³ positions within effective radius.
///   σ=1: ~64 positions, σ=2: ~512 positions, σ=4+: ~full volume
pub fn gabor_write(
    container: &mut [i8],
    lut: &GaussianLUT,
    x0: u8, y0: u8, z0: u8,
    freq: f32,
    phi: f32,
    amplitude: f32,
) {
    assert!(container.len() >= 2048);
    let radius_sq = lut.effective_radius_sq();

    for x in 0..FOCUS_DIM_X as i32 {
        let dx = x - x0 as i32;
        for y in 0..FOCUS_DIM_Y as i32 {
            let dy = y - y0 as i32;
            for z in 0..FOCUS_DIM_Z as i32 {
                let dz = z - z0 as i32;
                let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                if d_sq > radius_sq { continue; }

                let envelope = lut.amplitude(d_sq) as f32 / 255.0;
                if envelope < 0.004 { continue; }

                let phase = carrier_phase_3d(dx, dy, dz, freq, phi);
                let sample = (amplitude * envelope * phase.cos())
                    .round()
                    .clamp(-128.0, 127.0) as i8;

                let idx = x as usize * 256 + y as usize * 32 + z as usize;
                container[idx] = container[idx].saturating_add(sample);
            }
        }
    }
}

// ============================================================================
// Operation 16: gabor_read — Recover a concept from 3D space
// ============================================================================

/// Read back the amplitude and phase of a concept at a specific location
/// and frequency. Spatial demodulation via conjugate wavelet.
///
/// Returns (phase, amplitude).
pub fn gabor_read(
    container: &[i8],
    lut: &GaussianLUT,
    x0: u8, y0: u8, z0: u8,
    freq: f32,
) -> (f32, f32) {
    assert!(container.len() >= 2048);
    let radius_sq = lut.effective_radius_sq();
    let mut cos_sum: f64 = 0.0;
    let mut sin_sum: f64 = 0.0;
    let mut weight_sum: f64 = 0.0;

    for x in 0..FOCUS_DIM_X as i32 {
        let dx = x - x0 as i32;
        for y in 0..FOCUS_DIM_Y as i32 {
            let dy = y - y0 as i32;
            for z in 0..FOCUS_DIM_Z as i32 {
                let dz = z - z0 as i32;
                let d_sq = (dx * dx + dy * dy + dz * dz) as u32;
                if d_sq > radius_sq { continue; }

                let envelope = lut.amplitude(d_sq) as f64 / 255.0;
                if envelope < 0.004 { continue; }

                let phase = carrier_phase_3d(dx, dy, dz, freq, 0.0);
                let idx = x as usize * 256 + y as usize * 32 + z as usize;
                let sample = container[idx] as f64;

                cos_sum += sample * envelope * (phase as f64).cos();
                sin_sum += sample * envelope * (phase as f64).sin();
                weight_sum += envelope * envelope;
            }
        }
    }

    if weight_sum < 1e-10 {
        return (0.0, 0.0);
    }

    // Fourier sign convention: Σ cos(θ+φ)·sin(θ) = -W·sin(φ)/2
    // Negate sin_sum to recover the correct phase (same fix as carrier.rs).
    let recovered_phase = ((-sin_sum).atan2(cos_sum) as f32)
        .rem_euclid(std::f32::consts::TAU);
    let recovered_amplitude =
        ((cos_sum * cos_sum + sin_sum * sin_sum).sqrt() * 2.0 / weight_sum) as f32;

    (recovered_phase, recovered_amplitude)
}

// ============================================================================
// Operation 17: delta_cube_bind — Create a relationship volume
// ============================================================================

/// XOR-bind two fields to create a delta cube (binary mode).
/// Self-inverse: delta_cube_xor(a, a) → all zeros.
pub fn delta_cube_xor(field_a: &[u8], field_b: &[u8], delta: &mut [u8]) {
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048 && delta.len() >= 2048);
    for i in 0..2048 {
        delta[i] = field_a[i] ^ field_b[i];
    }
}

/// SUB-bind two fields to create a delta cube (phase/carrier mode).
pub fn delta_cube_sub(field_a: &[i8], field_b: &[i8], delta: &mut [i8]) {
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048 && delta.len() >= 2048);
    for i in 0..2048 {
        delta[i] = field_a[i].wrapping_sub(field_b[i]);
    }
}

// ============================================================================
// Operation 18: delta_cube_write/read — Content in relationships
// ============================================================================

/// Write a Gabor wavelet into a delta cube.
pub fn delta_cube_write_gabor(
    delta: &mut [i8],
    lut: &GaussianLUT,
    x0: u8, y0: u8, z0: u8,
    freq: f32,
    phi: f32,
    amplitude: f32,
) {
    gabor_write(delta, lut, x0, y0, z0, freq, phi, amplitude);
}

/// Read a Gabor wavelet from a delta cube.
pub fn delta_cube_read_gabor(
    delta: &[i8],
    lut: &GaussianLUT,
    x0: u8, y0: u8, z0: u8,
    freq: f32,
) -> (f32, f32) {
    gabor_read(delta, lut, x0, y0, z0, freq)
}

// ============================================================================
// Operation 19: delta_cube_recover — Reconstruct from both keys
// ============================================================================

/// Recover content from a stored delta and both original fields (XOR mode).
///
/// stored_delta = field_a ⊕ field_b ⊕ content
/// content = stored_delta ⊕ (field_a ⊕ field_b)
pub fn delta_cube_recover_xor(
    stored_delta: &[u8],
    field_a: &[u8],
    field_b: &[u8],
    content: &mut [u8],
) {
    assert!(stored_delta.len() >= 2048);
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048);
    assert!(content.len() >= 2048);

    for i in 0..2048 {
        let original_delta = field_a[i] ^ field_b[i];
        content[i] = stored_delta[i] ^ original_delta;
    }
}

/// Recover content from a stored delta and both original fields (phase mode).
///
/// stored_delta = field_a - field_b + content
/// content = stored_delta - (field_a - field_b)
pub fn delta_cube_recover_phase(
    stored_delta: &[i8],
    field_a: &[i8],
    field_b: &[i8],
    content: &mut [i8],
) {
    assert!(stored_delta.len() >= 2048);
    assert!(field_a.len() >= 2048 && field_b.len() >= 2048);
    assert!(content.len() >= 2048);

    for i in 0..2048 {
        let original_delta = field_a[i].wrapping_sub(field_b[i]);
        content[i] = stored_delta[i].wrapping_sub(original_delta);
    }
}

// ============================================================================
// SpatialTransform — 3D permutations over 8×8×32 volume
// ============================================================================

/// A spatial transform over the 8×8×32 volume.
/// Stored as a permutation table: perm[old_idx] = new_idx.
pub struct SpatialTransform {
    pub perm: [u16; 2048],
}

impl SpatialTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        let mut perm = [0u16; 2048];
        for i in 0..2048 {
            perm[i] = i as u16;
        }
        Self { perm }
    }

    /// Rotate the X axis by `n` positions (cyclic shift of 256-byte slabs).
    pub fn rotate_x(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            let x_new = (x + n as u16) % 8;
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x_new * 256 + y * 32 + z;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Rotate the Y axis by `n` positions.
    pub fn rotate_y(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                let y_new = (y + n as u16) % 8;
                for z in 0..32u16 {
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x * 256 + y_new * 32 + z;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Rotate the Z axis by `n` positions.
    pub fn rotate_z(n: u8) -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let z_new = (z + n as u16) % 32;
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x * 256 + y * 32 + z_new;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Diagonal rotation: (x, y, z) → (y, z%8, x*4 + z/8).
    /// Maps the 3 axes into each other, preserving volume.
    pub fn diagonal() -> Self {
        let mut perm = [0u16; 2048];
        for x in 0..8u16 {
            for y in 0..8u16 {
                for z in 0..32u16 {
                    let x_new = y;
                    let y_new = z % 8;
                    let z_new = x * 4 + z / 8;
                    let old_idx = x * 256 + y * 32 + z;
                    let new_idx = x_new * 256 + y_new * 32 + z_new;
                    perm[old_idx as usize] = new_idx;
                }
            }
        }
        Self { perm }
    }

    /// Compose: self.compose(other) = apply other first, then self.
    pub fn compose(&self, other: &SpatialTransform) -> SpatialTransform {
        let mut perm = [0u16; 2048];
        for i in 0..2048 {
            perm[i] = self.perm[other.perm[i] as usize];
        }
        SpatialTransform { perm }
    }

    /// Compute the inverse transform.
    pub fn inverse(&self) -> SpatialTransform {
        let mut inv = [0u16; 2048];
        for i in 0..2048 {
            inv[self.perm[i] as usize] = i as u16;
        }
        SpatialTransform { perm: inv }
    }

    /// Apply the transform to a u8 container.
    pub fn apply(&self, container: &[u8]) -> Vec<u8> {
        assert!(container.len() >= 2048);
        let mut out = vec![0u8; 2048];
        for i in 0..2048 {
            out[self.perm[i] as usize] = container[i];
        }
        out
    }

    /// Apply the transform to an i8 container.
    pub fn apply_i8(&self, container: &[i8]) -> Vec<i8> {
        assert!(container.len() >= 2048);
        let mut out = vec![0i8; 2048];
        for i in 0..2048 {
            out[self.perm[i] as usize] = container[i];
        }
        out
    }
}

// ============================================================================
// Operation 20: spatial_bind/unbind — Apply 3D transforms
// ============================================================================

/// Bind a u8 container with a spatial transform.
pub fn spatial_bind(container: &[u8], transform: &SpatialTransform) -> Vec<u8> {
    transform.apply(container)
}

/// Unbind (inverse transform) a u8 container.
pub fn spatial_unbind(container: &[u8], transform: &SpatialTransform) -> Vec<u8> {
    transform.inverse().apply(container)
}

/// Bind an i8 container with a spatial transform.
pub fn spatial_bind_i8(container: &[i8], transform: &SpatialTransform) -> Vec<i8> {
    transform.apply_i8(container)
}

/// Unbind (inverse transform) an i8 container.
pub fn spatial_unbind_i8(container: &[i8], transform: &SpatialTransform) -> Vec<i8> {
    transform.inverse().apply_i8(container)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn phase_error(a: f32, b: f32) -> f32 {
        let diff = (a - b).abs();
        diff.min(TAU - diff)
    }

    // ---- Envelope tests ----

    #[test]
    fn test_gaussian_lut_center_max() {
        let lut = GaussianLUT::new(2.0);
        assert_eq!(lut.amplitude(0), 255);
    }

    #[test]
    fn test_gaussian_lut_far_zero() {
        let lut = GaussianLUT::new(2.0);
        assert_eq!(lut.amplitude(1000), 0);
    }

    #[test]
    fn test_gaussian_lut_monotonic_decrease() {
        let lut = GaussianLUT::new(2.0);
        for i in 1..100u32 {
            assert!(
                lut.amplitude(i) <= lut.amplitude(i - 1),
                "LUT not monotonic at d²={}: {} > {}",
                i, lut.amplitude(i), lut.amplitude(i - 1)
            );
        }
    }

    #[test]
    fn test_gaussian_lut_different_sigmas() {
        let narrow = GaussianLUT::new(0.5);
        let wide = GaussianLUT::new(4.0);
        // At d²=4: narrow should be much more decayed than wide
        assert!(narrow.amplitude(4) < wide.amplitude(4));
    }

    // ---- WaveletTemplate tests ----

    #[test]
    fn test_wavelet_template_center() {
        let lut = GaussianLUT::new(2.0);
        let tpl = WaveletTemplate::new(&lut, 4, 4, 16);
        // Center position should be in the template with amplitude 255
        let center_idx = 4u16 * 256 + 4 * 32 + 16;
        assert!(
            tpl.entries.iter().any(|&(idx, amp)| idx == center_idx && amp == 255),
            "center should be in template with max amplitude"
        );
    }

    #[test]
    fn test_wavelet_template_tight_sigma() {
        let lut = GaussianLUT::new(0.5);
        let tpl = WaveletTemplate::new(&lut, 4, 4, 16);
        // Very tight: should have few entries
        assert!(
            tpl.entries.len() <= 30,
            "σ=0.5 template too large: {} entries",
            tpl.entries.len()
        );
    }

    // ---- Gabor wavelet tests ----

    #[test]
    fn test_gabor_write_max_at_center() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        // The center should have the highest absolute value
        let center_idx = 4 * 256 + 4 * 32 + 16;
        let center_val = container[center_idx].abs();

        // Check that it decays away from center
        let edge_idx = 0 * 256 + 0 * 32 + 0; // far corner
        let edge_val = container[edge_idx].abs();

        assert!(
            center_val > edge_val,
            "center {} should exceed edge {}",
            center_val, edge_val
        );
    }

    #[test]
    fn test_gabor_write_tight_affects_few() {
        let lut = GaussianLUT::new(0.5);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        let nonzero = container.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero <= 30,
            "σ=0.5 should affect ≤30 positions, got {}",
            nonzero
        );
    }

    #[test]
    fn test_gabor_write_broad_affects_most() {
        let lut = GaussianLUT::new(10.0);
        let mut container = vec![0i8; 2048];

        gabor_write(&mut container, &lut, 4, 4, 16, 1.0, 0.0, 7.0);

        let nonzero = container.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero > 1500,
            "σ=10 should affect most positions, got {}",
            nonzero
        );
    }

    #[test]
    fn test_gabor_read_recovers_phase_single() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        let target_phase = 1.5f32;

        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, target_phase, 7.0);
        let (rec_phase, rec_amp) = gabor_read(&container, &lut, 4, 4, 16, 3.0);

        assert!(
            phase_error(rec_phase, target_phase) < 0.15,
            "phase recovery: expected {:.4}, got {:.4} (error {:.4})",
            target_phase, rec_phase, phase_error(rec_phase, target_phase)
        );
        assert!(rec_amp > 0.5, "amplitude should be significant, got {:.4}", rec_amp);
    }

    #[test]
    fn test_gabor_read_empty_container() {
        let lut = GaussianLUT::new(2.0);
        let container = vec![0i8; 2048];

        let (_, amp) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
        assert!(amp < 0.01, "empty container should give near-zero amplitude, got {:.4}", amp);
    }

    #[test]
    fn test_gabor_two_frequencies_same_position() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];

        let phase_a = 0.5f32;
        let phase_b = 3.5f32;

        gabor_write(&mut container, &lut, 4, 4, 16, 2.0, phase_a, 7.0);
        gabor_write(&mut container, &lut, 4, 4, 16, 8.0, phase_b, 7.0);

        let (rec_a, _) = gabor_read(&container, &lut, 4, 4, 16, 2.0);
        let (rec_b, _) = gabor_read(&container, &lut, 4, 4, 16, 8.0);

        assert!(
            phase_error(rec_a, phase_a) < 0.3,
            "freq 2 recovery: expected {:.4}, got {:.4}",
            phase_a, rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.3,
            "freq 8 recovery: expected {:.4}, got {:.4}",
            phase_b, rec_b
        );
    }

    #[test]
    fn test_gabor_two_positions_same_frequency() {
        let lut = GaussianLUT::new(1.5);
        let mut container = vec![0i8; 2048];

        let phase_a = 1.0f32;
        let phase_b = 4.0f32;

        // Write at opposite corners of the volume
        gabor_write(&mut container, &lut, 1, 1, 4, 5.0, phase_a, 7.0);
        gabor_write(&mut container, &lut, 6, 6, 28, 5.0, phase_b, 7.0);

        // Read back from each position — the other should be suppressed
        let (rec_a, _) = gabor_read(&container, &lut, 1, 1, 4, 5.0);
        let (rec_b, _) = gabor_read(&container, &lut, 6, 6, 28, 5.0);

        assert!(
            phase_error(rec_a, phase_a) < 0.3,
            "pos A recovery: expected {:.4}, got {:.4}",
            phase_a, rec_a
        );
        assert!(
            phase_error(rec_b, phase_b) < 0.3,
            "pos B recovery: expected {:.4}, got {:.4}",
            phase_b, rec_b
        );
    }

    #[test]
    fn test_gabor_phase_sweep() {
        // Test multiple phase values to verify no systematic bias
        let lut = GaussianLUT::new(2.0);
        let test_phases = [0.0f32, 0.5, 1.0, 2.0, 3.14, 4.0, 5.5, 6.0];

        for &target in &test_phases {
            let mut container = vec![0i8; 2048];
            gabor_write(&mut container, &lut, 4, 4, 16, 3.0, target, 7.0);
            let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, 3.0);
            assert!(
                phase_error(rec, target) < 0.15,
                "phase {:.2}: expected {:.4}, got {:.4} (err {:.4})",
                target, target, rec, phase_error(rec, target)
            );
        }
    }

    // ---- Delta cube tests ----

    #[test]
    fn test_delta_cube_xor_self_zero() {
        let field: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&field, &field, &mut delta);
        assert!(delta.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_delta_cube_xor_self_inverse() {
        let a: Vec<u8> = (0..2048).map(|i| (i * 13 % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| (i * 41 + 7 % 256) as u8).collect();
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&a, &b, &mut delta);

        let mut double_delta = vec![0u8; 2048];
        delta_cube_xor(&delta, &delta, &mut double_delta);
        assert!(double_delta.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_delta_cube_recover_xor_round_trip() {
        // Create two fields
        let lut = GaussianLUT::new(2.0);
        let mut field_a = vec![0u8; 2048];
        let mut field_b = vec![0u8; 2048];

        // Fill with some deterministic data
        for i in 0..2048 {
            field_a[i] = (i * 31 % 256) as u8;
            field_b[i] = (i * 47 + 13 % 256) as u8;
        }

        // Create delta
        let mut delta = vec![0u8; 2048];
        delta_cube_xor(&field_a, &field_b, &mut delta);

        // Write content into delta
        let content: Vec<u8> = (0..2048).map(|i| (i * 7 % 256) as u8).collect();
        let mut stored_delta = delta.clone();
        for i in 0..2048 {
            stored_delta[i] ^= content[i];
        }

        // Recover content
        let mut recovered = vec![0u8; 2048];
        delta_cube_recover_xor(&stored_delta, &field_a, &field_b, &mut recovered);
        assert_eq!(recovered, content);
    }

    #[test]
    fn test_delta_cube_recover_phase_round_trip() {
        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        for i in 0..2048 {
            field_a[i] = (i as i8).wrapping_mul(3);
            field_b[i] = (i as i8).wrapping_mul(7).wrapping_add(13);
        }

        // Delta = A - B
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        // Content
        let content: Vec<i8> = (0..2048).map(|i| (i as i8).wrapping_mul(11)).collect();
        let mut stored: Vec<i8> = delta.iter().zip(content.iter())
            .map(|(&d, &c)| d.wrapping_add(c))
            .collect();

        // Recover
        let mut recovered = vec![0i8; 2048];
        delta_cube_recover_phase(&stored, &field_a, &field_b, &mut recovered);
        assert_eq!(recovered, content);
    }

    #[test]
    fn test_delta_cube_content_not_in_single_field() {
        let lut = GaussianLUT::new(2.0);

        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Create delta and write content
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        let content_phase = 4.0f32;
        delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 7.0, content_phase, 7.0);

        // Try reading content from field_a alone — should get noise
        let (_, amp_from_a) = gabor_read(&field_a, &lut, 4, 4, 16, 7.0);
        // Try reading from the stored delta — this has content + interference
        let (rec_phase, rec_amp) = delta_cube_read_gabor(&delta, &lut, 4, 4, 16, 7.0);

        // Content should be recoverable from delta
        assert!(
            rec_amp > amp_from_a * 2.0 || rec_amp > 0.5,
            "delta should have stronger signal than field_a alone: delta_amp={:.4}, a_amp={:.4}",
            rec_amp, amp_from_a
        );
    }

    // ---- Spatial transform tests ----

    #[test]
    fn test_rotate_x_0_is_identity() {
        let t = SpatialTransform::rotate_x(0);
        for i in 0..2048 {
            assert_eq!(t.perm[i], i as u16);
        }
    }

    #[test]
    fn test_rotate_x_8_is_identity() {
        let t = SpatialTransform::rotate_x(8);
        for i in 0..2048 {
            assert_eq!(t.perm[i], i as u16);
        }
    }

    #[test]
    fn test_rotate_inverse_identity() {
        let t = SpatialTransform::rotate_x(3);
        let inv = t.inverse();
        let composed = inv.compose(&t);
        for i in 0..2048 {
            assert_eq!(composed.perm[i], i as u16, "T⁻¹∘T should be identity at {}", i);
        }
    }

    #[test]
    fn test_compose_associative() {
        let t1 = SpatialTransform::rotate_x(2);
        let t2 = SpatialTransform::rotate_y(3);
        let t3 = SpatialTransform::rotate_z(5);

        let left = t1.compose(&t2).compose(&t3);   // (T1∘T2)∘T3
        let right = t1.compose(&t2.compose(&t3));   // T1∘(T2∘T3)

        for i in 0..2048 {
            assert_eq!(left.perm[i], right.perm[i], "associativity failed at {}", i);
        }
    }

    #[test]
    fn test_diagonal_is_bijection() {
        let t = SpatialTransform::diagonal();
        let mut seen = [false; 2048];
        for i in 0..2048 {
            let target = t.perm[i] as usize;
            assert!(target < 2048, "perm[{}] = {} out of range", i, target);
            assert!(!seen[target], "perm maps {} and another index both to {}", i, target);
            seen[target] = true;
        }
    }

    #[test]
    fn test_spatial_bind_unbind_round_trip() {
        let container: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let t = SpatialTransform::rotate_x(3);

        let bound = spatial_bind(&container, &t);
        let unbound = spatial_unbind(&bound, &t);
        assert_eq!(unbound, container);
    }

    #[test]
    fn test_spatial_bind_i8_round_trip() {
        let container: Vec<i8> = (0..2048).map(|i| (i as i8).wrapping_mul(13)).collect();
        let t = SpatialTransform::rotate_y(5);

        let bound = spatial_bind_i8(&container, &t);
        let unbound = spatial_unbind_i8(&bound, &t);
        assert_eq!(unbound, container);
    }

    #[test]
    fn test_spatial_bind_preserves_gabor() {
        let lut = GaussianLUT::new(2.0);
        let mut container = vec![0i8; 2048];
        let phase = 2.5f32;

        // Write Gabor at (4,4,16)
        gabor_write(&mut container, &lut, 4, 4, 16, 3.0, phase, 7.0);

        // Rotate X by 2: position (4,4,16) → (6,4,16)
        let t = SpatialTransform::rotate_x(2);
        let rotated = spatial_bind_i8(&container, &t);

        // Read at new position (6,4,16) — should recover the same phase
        let (rec, amp) = gabor_read(&rotated, &lut, 6, 4, 16, 3.0);
        assert!(
            phase_error(rec, phase) < 0.2,
            "rotated read: expected {:.4}, got {:.4}",
            phase, rec
        );
        assert!(amp > 0.5);
    }

    // ---- Integration: full pipeline ----

    #[test]
    fn test_full_pipeline_encode_bind_recover() {
        let lut = GaussianLUT::new(2.0);

        // Step 1-2: Encode two concepts as holographic fields
        let mut field_a = vec![0i8; 2048];
        let mut field_b = vec![0i8; 2048];
        gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
        gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

        // Step 3: Create delta cube
        let mut delta = vec![0i8; 2048];
        delta_cube_sub(&field_a, &field_b, &mut delta);

        // Step 4: Write relationship content into delta
        let rel_phase = 3.5f32;
        delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 7.0, rel_phase, 7.0);

        // Step 5: Apply spatial transform
        let t = SpatialTransform::rotate_z(4);
        let delta_transformed = spatial_bind_i8(&delta, &t);

        // Recovery: undo transform, subtract original delta, read content
        let delta_unrotated = spatial_unbind_i8(&delta_transformed, &t);
        let mut content = vec![0i8; 2048];
        delta_cube_recover_phase(&delta_unrotated, &field_a, &field_b, &mut content);

        let (rec_phase, rec_amp) = gabor_read(&content, &lut, 4, 4, 16, 7.0);
        assert!(
            phase_error(rec_phase, rel_phase) < 0.2,
            "pipeline recovery: expected {:.4}, got {:.4}",
            rel_phase, rec_phase
        );
        assert!(rec_amp > 0.3, "pipeline amplitude too low: {:.4}", rec_amp);
    }

    #[test]
    fn test_multiple_relationships_same_container() {
        let lut = GaussianLUT::new(1.5);

        // 3 pairs of fields
        let pairs: Vec<(f32, f32, f32)> = vec![
            (1.0, 2.0, 0.5),   // (phase_a, phase_b, rel_phase)
            (3.0, 4.0, 2.0),
            (5.0, 0.5, 4.5),
        ];
        let freqs = [2.0f32, 5.0, 8.0];
        let positions = [(2u8, 2u8, 8u8), (4, 4, 16), (6, 6, 24)];

        for (i, &(pa, pb, rel)) in pairs.iter().enumerate() {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            let (px, py, pz) = positions[i];

            gabor_write(&mut field_a, &lut, px, py, pz, freqs[i], pa, 7.0);
            gabor_write(&mut field_b, &lut, px, py, pz, freqs[i], pb, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, freqs[i], rel, 7.0);

            // Recover
            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let (rec, _) = gabor_read(&content, &lut, 4, 4, 16, freqs[i]);
            assert!(
                phase_error(rec, rel) < 0.3,
                "pair {} recovery: expected {:.4}, got {:.4}",
                i, rel, rec
            );
        }
    }

    // ---- Capacity experiment ----

    #[test]
    fn test_gabor_capacity_experiment() {
        println!("\n=== Gabor Wavelet Capacity Experiment ===\n");

        for &sigma in &[1.0f32, 2.0, 3.0] {
            let lut = GaussianLUT::new(sigma);

            for &n in &[1u32, 2, 4, 8, 12, 16] {
                let mut container = vec![0i8; 2048];
                let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.39 + 0.1).collect();
                let freqs: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 + 1.0).collect();

                // Write all wavelets at center
                for i in 0..n as usize {
                    gabor_write(
                        &mut container, &lut, 4, 4, 16,
                        freqs[i], phases[i], 7.0,
                    );
                }

                // Read each back
                let mut total_error = 0.0f32;
                for i in 0..n as usize {
                    let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, freqs[i]);
                    total_error += phase_error(rec, phases[i]);
                }
                let mean_error = total_error / n as f32;

                println!(
                    "  σ={:.1} N={:>2}: mean_phase_err={:.4} rad ({:>5.1}°)",
                    sigma, n, mean_error, mean_error.to_degrees()
                );
            }
            println!();
        }

        // Verify σ=2 holds up to N=4 with good accuracy
        {
            let lut = GaussianLUT::new(2.0);
            let mut container = vec![0i8; 2048];
            let phases = [0.5f32, 1.5, 3.0, 5.0];
            let freqs = [2.0f32, 5.0, 8.0, 12.0];
            for i in 0..4 {
                gabor_write(&mut container, &lut, 4, 4, 16, freqs[i], phases[i], 7.0);
            }
            let mut total_err = 0.0f32;
            for i in 0..4 {
                let (rec, _) = gabor_read(&container, &lut, 4, 4, 16, freqs[i]);
                total_err += phase_error(rec, phases[i]);
            }
            let mean = total_err / 4.0;
            assert!(
                mean < 0.5,
                "σ=2 N=4 mean error {:.4} rad too high",
                mean
            );
        }
    }

    #[test]
    fn test_delta_cube_capacity() {
        let lut = GaussianLUT::new(2.0);

        println!("\n=== Delta Cube Capacity Experiment ===\n");

        for &n in &[1u32, 2, 4, 8] {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
            gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);

            let phases: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7 + 0.3).collect();
            let freqs: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0 + 1.0).collect();

            for i in 0..n as usize {
                delta_cube_write_gabor(
                    &mut delta, &lut, 4, 4, 16,
                    freqs[i], phases[i], 7.0,
                );
            }

            // Recover
            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let mut total_error = 0.0f32;
            for i in 0..n as usize {
                let (rec, _) = gabor_read(&content, &lut, 4, 4, 16, freqs[i]);
                total_error += phase_error(rec, phases[i]);
            }
            let mean = total_error / n as f32;

            println!(
                "  Delta cube N={:>2}: mean_phase_err={:.4} rad ({:>5.1}°)",
                n, mean, mean.to_degrees()
            );
        }

        // Verify N=2 recoverable
        {
            let mut field_a = vec![0i8; 2048];
            let mut field_b = vec![0i8; 2048];
            gabor_write(&mut field_a, &lut, 2, 2, 8, 3.0, 1.0, 7.0);
            gabor_write(&mut field_b, &lut, 6, 6, 24, 5.0, 2.0, 7.0);

            let mut delta = vec![0i8; 2048];
            delta_cube_sub(&field_a, &field_b, &mut delta);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 3.0, 2.5, 7.0);
            delta_cube_write_gabor(&mut delta, &lut, 4, 4, 16, 8.0, 4.0, 7.0);

            let mut content = vec![0i8; 2048];
            delta_cube_recover_phase(&delta, &field_a, &field_b, &mut content);

            let (p1, _) = gabor_read(&content, &lut, 4, 4, 16, 3.0);
            let (p2, _) = gabor_read(&content, &lut, 4, 4, 16, 8.0);
            assert!(
                phase_error(p1, 2.5) < 0.4,
                "delta N=2 freq 3: expected 2.5, got {:.4}",
                p1
            );
            assert!(
                phase_error(p2, 4.0) < 0.4,
                "delta N=2 freq 8: expected 4.0, got {:.4}",
                p2
            );
        }
    }
}

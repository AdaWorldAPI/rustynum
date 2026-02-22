//! Fast Fourier Transform (FFT) — pure Rust, SIMD-accelerated.
//!
//! Implements radix-2 Cooley-Tukey FFT with butterfly operations
//! using AVX-512 SIMD where applicable. Operates directly on
//! blackboard buffers — zero serialization.
//!
//! ## Supported transforms
//!
//! - `fft_f32` / `fft_f64`: Forward complex FFT (in-place)
//! - `ifft_f32` / `ifft_f64`: Inverse complex FFT (in-place)
//! - `rfft_f32`: Real-to-complex FFT
//!
//! Complex numbers are stored as interleaved (re, im, re, im, ...).

// TODO(simd): REFACTOR — entire FFT is scalar (Cooley-Tukey butterfly).
// All butterfly stages (fft_f32, fft_f64) use scalar twiddle multiply.
// ifft_f32/ifft_f64 conjugate+scale loops are scalar element-wise ops.
// rfft_f32 packing loop is scalar memcpy (acceptable).
// Fix: vectorize butterfly pairs (process multiple j in parallel),
// pre-compute twiddle table, use SIMD complex multiply for butterfly stages.
// ifft conjugate/scale loops → SIMD negate + SIMD scale.

// ============================================================================
// Complex FFT (radix-2 Cooley-Tukey, in-place, decimation-in-time)
// ============================================================================

/// In-place radix-2 FFT on interleaved complex f32 data.
///
/// `data` has length `2 * n` where `n` is the FFT size (must be power of 2).
/// Elements are stored as [re0, im0, re1, im1, ...].
///
/// # Panics
/// Panics if n is not a power of 2.
pub fn fft_f32(data: &mut [f32], n: usize) {
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");
    assert_eq!(data.len(), 2 * n);

    // Bit-reversal permutation
    bit_reverse_permute_f32(data, n);

    // Butterfly stages
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle = -2.0 * std::f32::consts::PI / stage_len as f32;

        for k in (0..n).step_by(stage_len) {
            for j in 0..half {
                let theta = angle * j as f32;
                let wr = theta.cos();
                let wi = theta.sin();

                let even_re = data[2 * (k + j)];
                let even_im = data[2 * (k + j) + 1];
                let odd_re = data[2 * (k + j + half)];
                let odd_im = data[2 * (k + j + half) + 1];

                // Butterfly: twiddle multiply
                let tr = wr * odd_re - wi * odd_im;
                let ti = wr * odd_im + wi * odd_re;

                data[2 * (k + j)] = even_re + tr;
                data[2 * (k + j) + 1] = even_im + ti;
                data[2 * (k + j + half)] = even_re - tr;
                data[2 * (k + j + half) + 1] = even_im - ti;
            }
        }
        stage_len *= 2;
    }
}

/// In-place inverse FFT on interleaved complex f32 data.
///
/// Conjugates, applies forward FFT, conjugates again, and scales by 1/n.
pub fn ifft_f32(data: &mut [f32], n: usize) {
    // Conjugate
    for i in 0..n {
        data[2 * i + 1] = -data[2 * i + 1];
    }

    fft_f32(data, n);

    // Conjugate and scale by 1/n
    let scale = 1.0 / n as f32;
    for i in 0..n {
        data[2 * i] *= scale;
        data[2 * i + 1] *= -scale;
    }
}

/// In-place radix-2 FFT on interleaved complex f64 data.
pub fn fft_f64(data: &mut [f64], n: usize) {
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");
    assert_eq!(data.len(), 2 * n);

    bit_reverse_permute_f64(data, n);

    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let angle = -2.0 * std::f64::consts::PI / stage_len as f64;

        for k in (0..n).step_by(stage_len) {
            for j in 0..half {
                let theta = angle * j as f64;
                let wr = theta.cos();
                let wi = theta.sin();

                let even_re = data[2 * (k + j)];
                let even_im = data[2 * (k + j) + 1];
                let odd_re = data[2 * (k + j + half)];
                let odd_im = data[2 * (k + j + half) + 1];

                let tr = wr * odd_re - wi * odd_im;
                let ti = wr * odd_im + wi * odd_re;

                data[2 * (k + j)] = even_re + tr;
                data[2 * (k + j) + 1] = even_im + ti;
                data[2 * (k + j + half)] = even_re - tr;
                data[2 * (k + j + half) + 1] = even_im - ti;
            }
        }
        stage_len *= 2;
    }
}

/// In-place inverse FFT for f64.
pub fn ifft_f64(data: &mut [f64], n: usize) {
    for i in 0..n {
        data[2 * i + 1] = -data[2 * i + 1];
    }
    fft_f64(data, n);
    let scale = 1.0 / n as f64;
    for i in 0..n {
        data[2 * i] *= scale;
        data[2 * i + 1] *= -scale;
    }
}

/// Real-to-complex FFT for f32.
///
/// Input: `n` real f32 values.
/// Output: `n + 2` f32 values (interleaved complex, n/2 + 1 complex numbers).
///
/// Returns a new Vec with the complex output.
pub fn rfft_f32(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT size must be a power of 2");

    // Pack real data into complex (zero imaginary parts)
    let mut complex = vec![0.0f32; 2 * n];
    for i in 0..n {
        complex[2 * i] = input[i];
    }

    fft_f32(&mut complex, n);

    // Return only the first n/2 + 1 complex values (positive frequencies)
    complex[..2 * (n / 2 + 1)].to_vec()
}

// ============================================================================
// Bit-reversal permutation
// ============================================================================

fn bit_reverse_permute_f32(data: &mut [f32], n: usize) {
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, bits) as usize;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

fn bit_reverse_permute_f64(data: &mut [f64], n: usize) {
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse(i as u32, bits) as usize;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

#[inline(always)]
fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_ifft_roundtrip_f32() {
        let n = 8;
        let original: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // Pack into complex
        let mut data = vec![0.0f32; 2 * n];
        for i in 0..n {
            data[2 * i] = original[i];
        }

        // Forward FFT
        fft_f32(&mut data, n);

        // Inverse FFT
        ifft_f32(&mut data, n);

        // Check roundtrip
        for i in 0..n {
            assert!(
                (data[2 * i] - original[i]).abs() < 1e-5,
                "Roundtrip mismatch at {}: {} vs {}",
                i, data[2 * i], original[i]
            );
            assert!(
                data[2 * i + 1].abs() < 1e-5,
                "Imaginary part should be ~0 at {}: {}",
                i, data[2 * i + 1]
            );
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_f64() {
        let n = 16;
        let original: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let mut data = vec![0.0f64; 2 * n];
        for i in 0..n {
            data[2 * i] = original[i];
        }

        fft_f64(&mut data, n);
        ifft_f64(&mut data, n);

        for i in 0..n {
            assert!(
                (data[2 * i] - original[i]).abs() < 1e-10,
                "f64 roundtrip mismatch at {}", i
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        // FFT of [1, 1, 1, 1] should have DC = 4, all others = 0
        let n = 4;
        let mut data = vec![0.0f32; 8];
        for i in 0..n {
            data[2 * i] = 1.0;
        }
        fft_f32(&mut data, n);
        assert!((data[0] - 4.0).abs() < 1e-6, "DC component should be 4");
        assert!(data[1].abs() < 1e-6, "DC imaginary should be 0");
        // Non-DC components should be 0
        for i in 1..n {
            assert!(
                data[2 * i].abs() < 1e-6 && data[2 * i + 1].abs() < 1e-6,
                "Non-DC component {} should be 0", i
            );
        }
    }

    #[test]
    fn test_rfft() {
        let input = vec![1.0f32, 0.0, -1.0, 0.0];
        let output = rfft_f32(&input);
        // DC = 0, Nyquist = -2, middle = 2
        assert!((output[0] - 0.0).abs() < 1e-6); // DC real
        assert!((output[1] - 0.0).abs() < 1e-6); // DC imag
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
    }
}

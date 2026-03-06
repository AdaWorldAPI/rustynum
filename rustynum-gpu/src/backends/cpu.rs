//! CPU compute backend — always available, zero additional dependencies.
//!
//! Wraps rustynum-core's SIMD dispatch:
//!   AVX-512 VPOPCNTDQ (Tiger Lake, Sapphire Rapids)
//!   → AVX2 Harley-Seal (Meteor Lake, most laptops)
//!   → scalar POPCNT (fallback)
//!
//! This is the baseline. Every other backend must beat it to be selected.

use crate::traits::*;
use rustynum_core::simd::{self, HdrResult, PreciseMode};

pub struct CpuBackend {
    info: BackendInfo,
}

impl CpuBackend {
    pub fn new() -> Self {
        let has_avx512 = cfg!(target_arch = "x86_64") && {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx512vpopcntdq")
                    && is_x86_feature_detected!("avx512f")
            }
            #[cfg(not(target_arch = "x86_64"))]
            false
        };

        let has_vnni = cfg!(target_arch = "x86_64") && {
            #[cfg(target_arch = "x86_64")]
            {
                is_x86_feature_detected!("avx512vnni")
                    || is_x86_feature_detected!("avx2") // AVX2-VNNI on Meteor Lake
            }
            #[cfg(not(target_arch = "x86_64"))]
            false
        };

        let name = if has_avx512 {
            "CPU AVX-512 VPOPCNTDQ"
        } else if has_vnni {
            "CPU AVX2+VNNI"
        } else {
            "CPU scalar"
        };

        // Rough INT8 TOPS estimate
        let int8_tops = if has_avx512 {
            2.0 // 6 P-cores × 512-bit × ~GHz
        } else if has_vnni {
            1.0
        } else {
            0.1
        };

        CpuBackend {
            info: BackendInfo {
                name: name.to_string(),
                device_kind: DeviceKind::Cpu,
                available_memory: sysinfo_total_ram(),
                unified_memory: true,
                int8_tops,
            },
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn info(&self) -> &BackendInfo {
        &self.info
    }

    fn estimate(&self, op: OpHint, batch_size: usize, _element_bytes: usize) -> f64 {
        match op {
            // Belichtungsmesser with AVX-512 is the gold standard
            OpHint::HammingDistance | OpHint::HammingPrefix => {
                if self.info.name.contains("AVX-512") {
                    200.0 * batch_size as f64 // the 200x factor
                } else {
                    1.0 * batch_size as f64 // AVX2 baseline
                }
            }
            OpHint::DotI8 => {
                if self.info.name.contains("VNNI") {
                    10.0 * batch_size as f64
                } else {
                    1.0 * batch_size as f64
                }
            }
            OpHint::MatmulF32 => 1.0 * batch_size as f64, // rustyblas baseline
            OpHint::ElementwiseF32 => 1.0 * batch_size as f64,
            OpHint::ReduceF32 => 1.0 * batch_size as f64,
        }
    }

    fn hamming_batch(
        &self,
        query: &[u8],
        database: &[u8],
        num_rows: usize,
        row_bytes: usize,
    ) -> Option<Vec<u64>> {
        Some(simd::hamming_batch(query, database, num_rows, row_bytes))
    }

    fn hdr_cascade_search(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        num_vectors: usize,
        threshold: u64,
        precise_mode: PreciseMode,
    ) -> Option<Vec<HdrResult>> {
        Some(simd::hdr_cascade_search(
            query,
            database,
            vec_bytes,
            num_vectors,
            threshold,
            precise_mode,
        ))
    }

    fn sgemm(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _beta: f32,
    ) -> bool {
        // TODO: delegate to rustyblas::sgemm
        false
    }
}

/// Best-effort total system RAM detection.
fn sysinfo_total_ram() -> usize {
    // Read /proc/meminfo on Linux, fallback to 0
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_detection() {
        let cpu = CpuBackend::new();
        println!("CPU backend: {:?}", cpu.info());
        assert_eq!(cpu.info().device_kind, DeviceKind::Cpu);
        assert!(cpu.info().unified_memory);
    }

    #[test]
    fn test_cpu_hamming_batch() {
        let cpu = CpuBackend::new();
        let query = vec![0xAA_u8; 128];
        let mut database = vec![0x55_u8; 128 * 10];
        // Make one candidate identical to query
        database[128 * 3..128 * 4].copy_from_slice(&query);

        let distances = cpu.hamming_batch(&query, &database, 10, 128).unwrap();
        assert_eq!(distances.len(), 10);
        assert_eq!(distances[3], 0); // identical
        assert!(distances[0] > 0); // different
    }
}

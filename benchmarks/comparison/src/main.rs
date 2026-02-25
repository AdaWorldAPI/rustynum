/// Head-to-head benchmark: ndarray vs manual (representing rustynum-style ops)
///
/// This benchmark measures ndarray performance on the same operations that
/// rustynum benchmarks, allowing apples-to-apples comparison.
use ndarray::prelude::*;
use std::time::Instant;

fn black_box<T>(x: T) -> T {
    unsafe { std::ptr::read_volatile(&x) }
}

fn bench<F: FnMut()>(name: &str, n_iter: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..n_iter / 10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..n_iter {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / n_iter as f64;
    println!("  {:<40} {:>10.2} ns/iter  ({} iters)", name, per_iter_ns, n_iter);
    per_iter_ns
}

fn main() {
    println!("=== ndarray vs rustynum: Head-to-Head Benchmark ===\n");

    // --- Element-wise addition ---
    println!("--- Element-wise Addition (f32) ---");
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    for &size in &sizes {
        let a = Array1::<f32>::ones(size);
        let b = Array1::<f32>::ones(size);
        let n_iter = if size <= 10_000 { 100_000 } else if size <= 100_000 { 10_000 } else { 1_000 };
        bench(&format!("ndarray add f32 [{}]", size), n_iter, || {
            let c = &a + &b;
            black_box(&c);
        });
    }

    // --- Dot product ---
    println!("\n--- Dot Product (f32) ---");
    for &size in &sizes {
        let a = Array1::<f32>::ones(size);
        let b = Array1::<f32>::ones(size);
        let n_iter = if size <= 10_000 { 100_000 } else if size <= 100_000 { 10_000 } else { 1_000 };
        bench(&format!("ndarray dot f32 [{}]", size), n_iter, || {
            let c = a.dot(&b);
            black_box(c);
        });
    }

    // --- Mean ---
    println!("\n--- Mean (f64) ---");
    for &size in &sizes {
        let a = Array1::<f64>::from_elem(size, 1.5);
        let n_iter = if size <= 10_000 { 100_000 } else if size <= 100_000 { 10_000 } else { 1_000 };
        bench(&format!("ndarray mean f64 [{}]", size), n_iter, || {
            let m = a.mean().unwrap();
            black_box(m);
        });
    }

    // --- Sum ---
    println!("\n--- Sum (f32) ---");
    for &size in &sizes {
        let a = Array1::<f32>::ones(size);
        let n_iter = if size <= 10_000 { 100_000 } else if size <= 100_000 { 10_000 } else { 1_000 };
        bench(&format!("ndarray sum f32 [{}]", size), n_iter, || {
            let s = a.sum();
            black_box(s);
        });
    }

    // --- Standard Deviation ---
    println!("\n--- Standard Deviation (f64) ---");
    for &size in &sizes {
        let a = Array1::<f64>::from_elem(size, 1.5);
        let n_iter = if size <= 10_000 { 50_000 } else if size <= 100_000 { 5_000 } else { 500 };
        bench(&format!("ndarray std f64 [{}]", size), n_iter, || {
            let s = a.std(1.0);
            black_box(s);
        });
    }

    // --- Matrix multiply (GEMM) ---
    println!("\n--- Matrix Multiply (f32 GEMM) ---");
    let gemm_sizes = [32, 64, 128, 256, 512, 1024];
    for &size in &gemm_sizes {
        let a = Array2::<f32>::ones((size, size));
        let b = Array2::<f32>::ones((size, size));
        let n_iter = if size <= 64 {
            10_000
        } else if size <= 256 {
            100
        } else if size <= 512 {
            10
        } else {
            5
        };
        let ns = bench(&format!("ndarray matmul f32 [{}x{}]", size, size), n_iter, || {
            let c = a.dot(&b);
            black_box(&c);
        });
        let flops = 2.0 * (size as f64).powi(3);
        let gflops = flops / ns;
        println!("    -> {:.2} GFLOPS", gflops);
    }

    // --- Matrix multiply (f64 GEMM) ---
    println!("\n--- Matrix Multiply (f64 GEMM) ---");
    for &size in &gemm_sizes {
        let a = Array2::<f64>::ones((size, size));
        let b = Array2::<f64>::ones((size, size));
        let n_iter = if size <= 64 {
            10_000
        } else if size <= 256 {
            100
        } else if size <= 512 {
            10
        } else {
            5
        };
        let ns = bench(&format!("ndarray matmul f64 [{}x{}]", size, size), n_iter, || {
            let c = a.dot(&b);
            black_box(&c);
        });
        let flops = 2.0 * (size as f64).powi(3);
        let gflops = flops / ns;
        println!("    -> {:.2} GFLOPS", gflops);
    }

    // --- Array construction ---
    println!("\n--- Array Construction ---");
    bench("ndarray zeros f32 [10000]", 100_000, || {
        let a = Array1::<f32>::zeros(10_000);
        black_box(&a);
    });
    bench("ndarray ones f32 [10000]", 100_000, || {
        let a = Array1::<f32>::ones(10_000);
        black_box(&a);
    });
    bench("ndarray linspace f64 [10000]", 10_000, || {
        let a = Array1::<f64>::linspace(0.0, 1.0, 10_000);
        black_box(&a);
    });

    // --- Slicing ---
    println!("\n--- Slicing ---");
    let big = Array1::<f32>::ones(100_000);
    bench("ndarray slice [100K -> 50K]", 1_000_000, || {
        let s = big.slice(s![..50_000]);
        black_box(&s);
    });

    // --- Transpose ---
    println!("\n--- Transpose ---");
    let mat = Array2::<f32>::ones((1000, 1000));
    bench("ndarray transpose [1000x1000]", 1_000_000, || {
        let t = mat.t();
        black_box(&t);
    });

    // --- Axis operations ---
    println!("\n--- Axis Reductions ---");
    let mat = Array2::<f64>::from_elem((1000, 100), 1.5);
    bench("ndarray sum axis=0 [1000x100]", 10_000, || {
        let s = mat.sum_axis(Axis(0));
        black_box(&s);
    });
    bench("ndarray sum axis=1 [1000x100]", 10_000, || {
        let s = mat.sum_axis(Axis(1));
        black_box(&s);
    });
    bench("ndarray mean axis=0 [1000x100]", 10_000, || {
        let s = mat.mean_axis(Axis(0)).unwrap();
        black_box(&s);
    });

    println!("\n=== Benchmark complete ===");
}

use rand::rngs::StdRng;
use rand::SeedableRng;
use rustynum_oracle::organic::*;
use rustynum_oracle::sweep::Base;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    println!("=== ORGANIC X-TRANS SWEEP ===\n");

    // Experiment 1: Recovery across D for key bases, channels=16, K=8
    println!("--- Exp 1: Recovery Error vs D (channels=16, K=8, no plasticity) ---");
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<10} {:<8}",
        "D", "Base", "MeanErr", "MeanAbsorp", "MinAbsorp", "MeanSim", "Flush"
    );
    for &d in &[1024usize, 2048, 4096, 8192] {
        for &base in &[
            Base::Binary,
            Base::Signed(5),
            Base::Signed(7),
            Base::Signed(9),
        ] {
            let r = measure_recovery_organic(d, base, 16, 8, false, &mut rng);
            println!(
                "{:<8} {:<12} {:<12.6} {:<12.4} {:<12.4} {:<10.4} {:?}",
                r.d,
                base_name(base),
                r.mean_error,
                r.mean_absorption,
                r.min_absorption,
                r.mean_similarity,
                r.flush_action
            );
        }
    }

    // Experiment 2: Recovery vs K (D=2048, channels=16)
    println!("\n--- Exp 2: Recovery Error vs K (D=2048, channels=16, Signed(5)) ---");
    println!(
        "{:<6} {:<12} {:<12} {:<12} {:<12} {:<10} {:<10}",
        "K", "MeanErr", "MeanAbsorp", "MinAbsorp", "MeanSim", "Flush", "Bits/C"
    );
    for &k in &[1, 3, 5, 8, 13, 16] {
        let r = measure_recovery_organic(2048, Base::Signed(5), 16, k, false, &mut rng);
        println!(
            "{:<6} {:<12.6} {:<12.4} {:<12.4} {:<12.4} {:<10?} {:<10.1}",
            k, r.mean_error, r.mean_absorption, r.min_absorption, r.mean_similarity,
            r.flush_action, r.bits_per_concept
        );
    }

    // Experiment 3: Effect of channels count
    println!("\n--- Exp 3: Effect of Channel Count (D=2048, K=8, Signed(5)) ---");
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<10} {:<10}",
        "Channels", "MeanErr", "MeanAbsorp", "MinAbsorp", "MeanSim", "PatQual"
    );
    for &ch in &[8, 16, 32, 64, 128] {
        let r = measure_recovery_organic(2048, Base::Signed(5), ch, 8, false, &mut rng);
        println!(
            "{:<10} {:<12.6} {:<12.4} {:<12.4} {:<10.4} {:<10}",
            ch, r.mean_error, r.mean_absorption, r.min_absorption,
            r.mean_similarity, r.pattern_quality
        );
    }

    // Experiment 4: Plasticity vs No Plasticity
    println!("\n--- Exp 4: Plasticity Effect (D=2048, channels=16, K=8) ---");
    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Plasticity", "Base", "MeanErr", "MeanAbsorp", "MeanSim"
    );
    for &base in &[Base::Signed(5), Base::Signed(9)] {
        for &plast in &[false, true] {
            let r =
                measure_recovery_organic(2048, base, 16, 8, plast, &mut rng);
            println!(
                "{:<12} {:<12} {:<12.6} {:<12.4} {:<12.4}",
                plast,
                base_name(base),
                r.mean_error,
                r.mean_absorption,
                r.mean_similarity
            );
        }
    }

    // Experiment 5: X-Trans Pattern Quality
    println!("\n--- Exp 5: X-Trans Pattern Quality ---");
    println!(
        "{:<8} {:<10} {:<10} {:<12} {:<12}",
        "D", "Channels", "MinDist", "Uniformity", "PosPerCh"
    );
    for &d in &[1024usize, 2048, 4096, 8192] {
        for &ch in &[8, 16, 32, 64] {
            let pat = XTransPattern::new(d, ch);
            println!(
                "{:<8} {:<10} {:<10} {:<12.4} {:<12}",
                d, ch, pat.min_same_channel_distance(), pat.size_uniformity(),
                pat.positions_per_channel
            );
        }
    }

    // Experiment 6: Head-to-head with standard sweep (same D, K, base)
    println!("\n--- Exp 6: Organic vs Standard (Signed(5), D=2048) ---");
    println!(
        "{:<8} {:<10} {:<12} {:<12} {:<12}",
        "K", "Method", "MeanErr", "Storage", "Bits/C"
    );
    for &k in &[1, 3, 5, 8, 13] {
        // Organic
        let org = measure_recovery_organic(2048, Base::Signed(5), 16, k, false, &mut rng);
        println!(
            "{:<8} {:<10} {:<12.6} {:<12} {:<12.1}",
            k, "organic", org.mean_error, org.storage_bytes, org.bits_per_concept
        );
        // Standard (from sweep module)
        let std_r = rustynum_oracle::sweep::measure_recovery(2048, Base::Signed(5), k, &mut rng);
        let std_bits = (2048.0 * 8.0) / k as f64;
        println!(
            "{:<8} {:<10} {:<12.6} {:<12} {:<12.1}",
            k, "standard", std_r.mean_error, 2048, std_bits
        );
    }

    // Experiment 7: Absorption decay as K grows
    println!("\n--- Exp 7: Absorption Decay vs K (D=4096, channels=32) ---");
    println!(
        "{:<6} {:<12} {:<12} {:<12} {:<10}",
        "K", "MeanAbsorp", "MinAbsorp", "MeanErr", "Flush"
    );
    for &k in &[1, 3, 5, 8, 13, 21, 32] {
        let r = measure_recovery_organic(4096, Base::Signed(5), 32, k, false, &mut rng);
        println!(
            "{:<6} {:<12.4} {:<12.4} {:<12.6} {:<10?}",
            k, r.mean_absorption, r.min_absorption, r.mean_error, r.flush_action
        );
    }

    // Experiment 8: Flush cycle test
    println!("\n--- Exp 8: Organic Flush Cycle (D=2048, channels=16, K=8) ---");
    let d = 2048;
    let channels = 16;
    let k = 8;
    let pattern = XTransPattern::new(d, channels);
    let mut wal = OrganicWAL::new(pattern);
    let mut container = vec![0i8; d];
    let plasticity_tracker = PlasticityTracker::new(k, 50);

    let templates = rustynum_oracle::sweep::generate_templates(
        k, d, Base::Signed(5), &mut rng,
    );
    for (i, t) in templates.iter().enumerate() {
        wal.register_concept(i as u32, t.clone());
    }

    // Write all concepts
    for i in 0..k {
        let r = wal.write(&mut container, i, 0.7, 0.1);
        println!("  Write concept {}: absorption={:.4}, channel={}", i, r.absorption, r.channel);
    }

    let coeffs_before: Vec<f32> = wal.coefficients.clone();
    println!("  Coefficients before flush: {:?}",
        coeffs_before.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>());

    // Flush
    let flush_result = organic_flush(&mut wal, &mut container, &plasticity_tracker, None);
    println!("  Flush: {} concepts rewritten, avg absorption={:.4}",
        flush_result.concepts_rewritten, flush_result.average_absorption);
    println!("  Extracted coefficients: {:?}",
        flush_result.coefficients_extracted.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>());
    println!("  Coefficients after flush: {:?}",
        wal.coefficients.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>());

    // Read back after flush
    let readbacks = wal.read_all(&container);
    println!("  Readback after flush:");
    for (id, sim, amp) in &readbacks {
        println!("    concept {}: similarity={:.4}, amplitude={:.4}", id, sim, amp);
    }

    // Flush with pruning
    println!("\n  --- Flush with pruning to top 4 ---");
    // Re-write concepts with varying amplitudes
    container.fill(0);
    for i in 0..k {
        let amp = if i < 4 { 0.8 } else { 0.1 };
        wal.write(&mut container, i, amp, 0.1);
    }
    let prune_result = organic_flush(&mut wal, &mut container, &plasticity_tracker, Some(4));
    println!("  Pruned concepts: {:?}", prune_result.concepts_pruned);
    println!("  Concepts rewritten: {}", prune_result.concepts_rewritten);
    println!("  Final coefficients: {:?}",
        wal.coefficients.iter().map(|c| format!("{:.4}", c)).collect::<Vec<_>>());
}

fn base_name(base: Base) -> &'static str {
    match base {
        Base::Binary => "Binary",
        Base::Unsigned(3) => "Uns(3)",
        Base::Unsigned(5) => "Uns(5)",
        Base::Unsigned(7) => "Uns(7)",
        Base::Signed(3) => "Sig(3)",
        Base::Signed(5) => "Sig(5)",
        Base::Signed(7) => "Sig(7)",
        Base::Signed(9) => "Sig(9)",
        _ => "Other",
    }
}

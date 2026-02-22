//! Absorption stress test and base-aware plasticity comparison.
//!
//! Finds the K breaking point for each configuration and compares
//! fixed vs base-aware plasticity scaling.
//!
//! Usage: cargo run --release --bin stress_test

fn main() {
    rustynum_oracle::run_absorption_stress_test();
    rustynum_oracle::run_plasticity_comparison();
}

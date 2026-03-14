# SESSION_C_CROSSPOLINATE.md

## Cross-pollinate 5 algorithms into rustynum from lance-graph

**Repo:** rustynum (WRITE)
**Read-only reference:** clone lance-graph for reading hdr.rs
**Prereq:** Session A completed (hdr.rs exists, simd renamed)
**Stop when:** `cargo test --workspace` passes, all 5 features have tests.

---

## SETUP

```bash
# Clone lance-graph read-only for reference
git clone https://github.com/AdaWorldAPI/lance-graph.git ../lance-graph-ref
# DO NOT write to lance-graph. Read only.

# Verify Session A completed
ls rustynum-core/src/hdr.rs          # must exist
ls rustynum-core/src/simd_avx512.rs  # must exist (renamed from simd_compat)
cargo test -p rustynum-core          # must pass
```

---

## FEATURE 1: ReservoirSample + Auto-Switch to Empirical Quantiles

**Read:** `../lance-graph-ref/crates/lance-graph/src/graph/blasgraph/hdr.rs`
**Find:** `struct ReservoirSample`, `fn skewness()`, `fn kurtosis()`, `fn quantile()`

**Copy into** `rustynum-core/src/hdr.rs`:

```rust
pub struct ReservoirSample {
    samples: Vec<u32>,
    capacity: usize,
    seen: u64,
}

impl ReservoirSample {
    pub fn new(capacity: usize) -> Self { ... }
    pub fn observe(&mut self, distance: u32) { ... }  // Vitter's Algorithm R
    pub fn quantile(&self, q: f32) -> u32 { ... }
    pub fn skewness(&self, mu: u32, sigma: u32) -> i32 { ... }  // Pearson's second
    pub fn kurtosis(&self, mu: u32, sigma: u32) -> u32 { ... }  // excess, ×100
    pub fn len(&self) -> usize { ... }
    fn fast_rand(seed: u64) -> u64 { ... }
}
```

**Add to Cascade struct:**

```rust
pub struct Cascade {
    // ...existing fields...
    reservoir: ReservoirSample,
    empirical_bands: [u32; 4],
    use_empirical: bool,
}
```

**Add auto-switch logic in `observe()`:**

```rust
if self.running_count % 1000 == 0 && self.reservoir.len() >= 100 {
    let skew = self.reservoir.skewness(self.mu, self.sigma);
    let kurt = self.reservoir.kurtosis(self.mu, self.sigma);
    let is_normal = skew.abs() < 2 && kurt > 200 && kurt < 500;

    if !is_normal {
        self.empirical_bands = [
            self.reservoir.quantile(0.001),  // Foveal
            self.reservoir.quantile(0.023),  // Near
            self.reservoir.quantile(0.159),  // Good
            self.reservoir.quantile(0.500),  // Weak
        ];
        self.use_empirical = true;
    } else {
        self.use_empirical = false;
    }
}
```

**Update `expose()` / `band()`:**

```rust
pub fn expose(&self, distance: u32) -> Band {
    let bands = if self.use_empirical { &self.empirical_bands } else { &self.bands };
    if distance < bands[0] { Band::Foveal }
    else if distance < bands[1] { Band::Near }
    else if distance < bands[2] { Band::Good }
    else if distance < bands[3] { Band::Weak }
    else { Band::Reject }
}
```

**Test:**

```rust
#[test]
fn reservoir_auto_switch_on_bimodal() {
    let mut cascade = Cascade::calibrate(&[8192; 100]); // normal start

    // Feed bimodal data: half near 4000, half near 8000
    for i in 0..2000 {
        let d = if i % 2 == 0 { 4000 + (i % 100) } else { 8000 + (i % 100) };
        cascade.observe(d);
    }

    // Should have switched to empirical
    assert!(cascade.use_empirical, "Bimodal data should trigger empirical mode");
}
```

---

## FEATURE 2: Integer Hot Path (replace f64 sigma)

**Read:** lance-graph hdr.rs `fn isqrt()` and `fn calibrate()`

**Copy `isqrt()`** into `rustynum-core/src/hdr.rs`:

```rust
pub fn isqrt(n: u32) -> u32 {
    if n == 0 { return 0; }
    let mut x = 1u32 << ((33 - n.leading_zeros()) / 2);
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x { return x; }
        x = x1;
    }
}
```

**Replace in `Cascade::calibrate()`:**

The current code uses f64 for sigma:
```rust
let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();
```

Replace with integer pre-computation:

```rust
// In calibrate(), compute bands as u32:
let sigma = isqrt(variance as u32).max(1);
let bands = [
    mu.saturating_sub(3 * sigma),  // Foveal: < μ - 3σ
    mu.saturating_sub(2 * sigma),  // Near:   < μ - 2σ
    mu.saturating_sub(sigma),      // Good:   < μ - σ
    mu,                            // Weak:   < μ
];
```

**Replace in `Cascade::query()` hot path:**

```rust
// BEFORE (float per query):
let s1_reject = threshold as f64 + 3.0 * sigma;
if (estimate as f64) <= s1_reject { ... }

// AFTER (integer, precomputed):
if projected < self.bands[2] { ... }  // one u32 compare. Done.
```

**Test:**

```rust
#[test]
fn isqrt_matches_f64_sqrt() {
    for n in [0, 1, 4, 9, 64, 4096, 8192, 65535, u32::MAX] {
        let integer = isqrt(n);
        let float = (n as f64).sqrt() as u32;
        assert!((integer as i64 - float as i64).abs() <= 1,
            "isqrt({}) = {}, f64 sqrt = {}", n, integer, float);
    }
}
```

---

## FEATURE 3: Persistent Calibration (kill per-query warmup)

**The change:** `Cascade::query()` no longer runs 128-sample warmup.
The calibration state persists across queries.

**Remove from `query()`:**

```rust
// DELETE this entire warmup block:
let warmup_n = 128.min(num_vectors);
let mut warmup_dists = Vec::with_capacity(warmup_n);
for i in 0..warmup_n {
    // ... warmup sampling ...
}
let var: f64 = { ... };
let sigma_pop = var.sqrt();
let sigma = sigma_est.max(sigma_pop).max(1.0);
```

**Replace with:**

```rust
// Use pre-calibrated bands. One u32 compare per candidate.
// No warmup. No f64. No per-query sigma estimation.
let s1_reject = self.bands[2];  // Good threshold = μ - σ
```

**The caller is responsible for calibration:**

```rust
// Application code:
let hdr = hdr::Cascade::calibrate(&initial_sample);  // ONCE

for query in queries {
    let hits = hdr.query(&query, &db, 10);  // no warmup inside
    for hit in &hits {
        hdr.observe(hit.distance);  // feed Welford
    }
    if let Some(shift) = hdr.drift() {
        hdr.recalibrate(&shift);  // only when distribution changes
    }
}
```

**Backward compat:** Keep the deprecated `hdr_cascade_search()` wrapper
in simd.rs. It creates a temporary Cascade with warmup-style calibration
so old callers still work:

```rust
#[deprecated]
pub fn hdr_cascade_search(...) -> Vec<hdr::RankedHit> {
    // Warmup calibration for backward compat
    let sample_dists = quick_warmup_sample(query, database, vec_bytes, 128);
    let cascade = hdr::Cascade::calibrate(&sample_dists);
    cascade.query(query, database, vec_bytes, num_vectors, precise_mode)
}
```

**Test:**

```rust
#[test]
fn persistent_calibration_no_warmup_overhead() {
    let db = random_database(10_000, 2048, 42);
    let query = random_vec(2048, 99);

    // Calibrate once
    let sample = pairwise_sample(&db, 2048, 200);
    let hdr = Cascade::calibrate(&sample);

    // Query 100 times — should not recalibrate internally
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = hdr.query(&query, &db, 2048, 10_000, PreciseMode::Off);
    }
    let persistent_time = start.elapsed();

    // Compare with old warmup-per-query approach
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = hdr_cascade_search(&query, &db, 2048, 10_000, 500, PreciseMode::Off);
    }
    let warmup_time = start.elapsed();

    println!("Persistent: {:?}", persistent_time);
    println!("Per-query warmup: {:?}", warmup_time);
    // Persistent should be faster (no 128 × 100 = 12800 wasted samples)
}
```

---

## FEATURE 4: Welford Shift Detection + Recalibrate Reset

**Read:** lance-graph hdr.rs `fn observe()`, `fn recalibrate()`

**Add Welford state to Cascade:**

```rust
pub struct Cascade {
    // ...existing fields...
    running_count: u64,
    running_mean: u64,
    running_m2: u64,
}
```

**Add `observe()` and `drift()`:**

```rust
impl Cascade {
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        let d = distance as u64;
        self.running_count += 1;
        self.reservoir.observe(distance);

        // Welford online update
        let delta = d as i64 - (self.running_mean / self.running_count.max(1)) as i64;
        self.running_mean += d;
        let new_mean = self.running_mean / self.running_count;
        let delta2 = d as i64 - new_mean as i64;
        self.running_m2 = self.running_m2.wrapping_add((delta.wrapping_mul(delta2)) as u64);

        // Check every 1000 observations
        if self.running_count % 1000 == 0 && self.running_count > 1000 {
            let running_mu = (self.running_mean / self.running_count) as u32;
            let running_var = (self.running_m2 / self.running_count) as u32;
            let running_sigma = isqrt(running_var).max(1);

            let mu_drift = running_mu.abs_diff(self.mu);
            let sigma_drift = running_sigma.abs_diff(self.sigma);

            if mu_drift > self.sigma / 2 || sigma_drift > self.sigma / 4 {
                return Some(ShiftAlert {
                    old_mu: self.mu,
                    new_mu: running_mu,
                    old_sigma: self.sigma,
                    new_sigma: running_sigma,
                    observations: self.running_count,
                });
            }
        }

        // Auto-switch check (from Feature 1)
        // ...skewness/kurtosis check here...

        None
    }

    pub fn drift(&self) -> Option<ShiftAlert> {
        // Read-only check without mutation. For callers that
        // separate observation from drift detection.
        if self.running_count < 2000 { return None; }
        let running_mu = (self.running_mean / self.running_count) as u32;
        let running_var = (self.running_m2 / self.running_count) as u32;
        let running_sigma = isqrt(running_var).max(1);
        let mu_drift = running_mu.abs_diff(self.mu);
        let sigma_drift = running_sigma.abs_diff(self.sigma);
        if mu_drift > self.sigma / 2 || sigma_drift > self.sigma / 4 {
            Some(ShiftAlert {
                old_mu: self.mu, new_mu: running_mu,
                old_sigma: self.sigma, new_sigma: running_sigma,
                observations: self.running_count,
            })
        } else { None }
    }

    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        self.mu = alert.new_mu;
        self.sigma = alert.new_sigma.max(1);
        self.bands = [
            self.mu.saturating_sub(3 * self.sigma),
            self.mu.saturating_sub(2 * self.sigma),
            self.mu.saturating_sub(self.sigma),
            self.mu,
        ];
        // RESET everything — clean slate
        self.running_count = 0;
        self.running_mean = 0;
        self.running_m2 = 0;
        self.reservoir = ReservoirSample::new(self.reservoir.capacity);
        self.use_empirical = false;
    }
}
```

**Test:**

```rust
#[test]
fn shift_detection_fires_on_distribution_change() {
    let mut hdr = Cascade::calibrate(&[8192; 200]); // μ=8192

    // Feed shifted distribution (μ=7500)
    let mut fired = false;
    for i in 0..5000 {
        let d = 7500 + (i % 100);
        if let Some(alert) = hdr.observe(d) {
            assert!(alert.new_mu < alert.old_mu);
            hdr.recalibrate(&alert);
            fired = true;
            break;
        }
    }
    assert!(fired, "Shift alert should fire");
    assert!(hdr.mu < 8000, "Recalibrated μ should reflect new distribution");
}
```

---

## FEATURE 5: Incremental Stroke 2 (from rustynum's own design)

This is already in rustynum's `hdr_cascade_search`. Verify it survived
the move to `Cascade::query()`. The pattern:

```rust
// Stroke 1: prefix distance
let d_prefix = hamming_fn(query_prefix, cand_prefix);

// Stroke 2: ONLY the remaining bytes (incremental, not full recompute)
let d_rest = hamming_fn(query_rest, cand_rest);
let d_full = d_prefix + d_rest;
```

If `Cascade::query()` lost the incremental pattern during refactoring,
restore it. If it's intact, add a test proving incrementality:

```rust
#[test]
fn stroke2_is_incremental() {
    let a = random_vec(2048, 1);
    let b = random_vec(2048, 2);
    let hamming = crate::simd::select_hamming_fn();

    let full = hamming(&a, &b);
    let split = 128; // 1/16
    let d_prefix = hamming(&a[..split], &b[..split]);
    let d_rest = hamming(&a[split..], &b[split..]);

    assert_eq!(full, d_prefix + d_rest, "Incremental must equal full");
}
```

---

## VERIFY

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --workspace
cargo clippy --workspace -- -D warnings

# Specifically test the new hdr module:
cargo test -p rustynum-core -- hdr

# Count: should have at least 5 new tests (one per feature)
cargo test -p rustynum-core -- hdr 2>&1 | grep "test result"
```

---

## NOT IN SCOPE

```
× Don't touch lance-graph (Session B already renamed it)
× Don't add PreciseMode to lance-graph (that's a lance-graph session)
× Don't add SIMD dispatch to lance-graph (comes with BitVec rebuild)
× Don't extract simd_avx2.rs from simd.rs (future session)
× Don't add GPU backends (roadmap)
× Don't refactor Plane/Node/Mask (separate prompt)
```

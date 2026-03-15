# L1_CACHE_BOUNDARY.md

## The 64KB Cliff: L1 Cache as Architectural Boundary

**Status:** Measured. Architectural constraint. Do not violate.

---

## THE MEASUREMENT

Popcount performance across buffer sizes on AVX-512 hardware:

```
SIZE      SIMD            SCALAR          RATIO       THROUGHPUT
──────────────────────────────────────────────────────────────────
8KB       74.65 ns        1.57 μs         21x FASTER  104 GiB/s
16KB      143.84 ns       3.07 μs         21x FASTER  104 GiB/s
32KB      284.14 ns       6.08 μs         21x FASTER  104 GiB/s
64KB      149.89 μs       12.37 μs        12x SLOWER  343 MiB/s
```

At 32KB: linear scaling, 104 GiB/s throughput. VPOPCNTDQ at full speed.
At 64KB: 528x throughput COLLAPSE. SIMD becomes 12x slower than scalar.

---

## ROOT CAUSE

64KB = L1d cache size on this CPU. The SIMD path reads at 104 GiB/s —
faster than L2 can feed L1. At 32KB everything fits. At 64KB every load
misses L1, pipeline stalls waiting for L2 (~4ns penalty per miss).

The scalar path is slower per-byte so it never outpaces the L2→L1 bandwidth.
It accidentally stays within the memory subsystem's comfort zone.

---

## ARCHITECTURAL CONSEQUENCE

```
SAFE ZONE:   buffer ≤ 32KB    → 21x SIMD speedup, 104 GiB/s
DANGER ZONE: buffer > 32KB    → performance cliff, SIMD SLOWER than scalar
DEAD ZONE:   buffer = 64KB    → 12x regression, 528x throughput drop
```

---

## VALIDATION OF PER-PLANE DESIGN

Our fixed container sizes:

```
SKU-16K:  16384 bits = 2048 bytes    ← SAFE (2KB, deep in L1)
SKU-64K:  65536 bits = 8192 bytes    ← SAFE (8KB, fits in L1)
Plane:    16384 bits = 2048 bytes    ← SAFE
Node SPO: 3 × 2048   = 6144 bytes   ← SAFE (6KB)
CogRecord: 4 × 2048  = 8192 bytes   ← SAFE (8KB)
```

ALL standard containers fit in L1. The 21x speedup applies to every
normal operation in the architecture.

---

## WHAT BREAKS

```
8 CogRecords concatenated:   8 × 8KB = 64KB   → CLIFF
16 Planes merged:            16 × 2KB = 32KB   → AT THE EDGE
32 Planes merged:            32 × 2KB = 64KB   → CLIFF
"Fingerprint the full batch": variable          → DANGEROUS
```

RULE: Never fingerprint/popcount/hamming across concatenated containers.
Always per-plane. The cascade already does this — stroke 1 reads 128 bytes
of ONE plane, not 128 bytes across a merged buffer.

---

## THE RULE

```
1. ALWAYS operate per-plane (2KB). NEVER merge planes for bulk operations.
2. Cascade strokes: 128B, 384B, 1536B — all deep in L1.
3. Batch operations: iterate over planes, not over concatenated buffers.
4. If you need distance over multiple planes (SPO node): accumulate
   per-plane distances, don't merge the data.
5. The 64KB boundary is not a bug to fix. It's a wall to respect.
```

---

## FOR FUTURE DESIGN

If anyone suggests:
- "Fingerprint the whole CogRecord as one 8KB vector" → OK (fits L1)
- "Fingerprint 8 records at once as 64KB" → NO (hits cliff)
- "4D CogRecord (4 × 16K = 64Kbit = 8KB)" → OK (8KB fits)
- "8D CogRecord (8 × 16K = 128Kbit = 16KB)" → OK (16KB fits)
- "32D CogRecord (32 × 16K = 512Kbit = 64KB)" → NO (hits cliff)

The L1 cache size determines the maximum useful container dimension.
At 2KB per plane: up to 16 planes = 32KB = safe.
Beyond that: per-plane iteration, never bulk.

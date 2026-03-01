# Architectural Rethink: The Concept Pipeline

## The Question

If we're serious about 3D SPO non-dilution, we need to rethink:
1. Whether the 90° vector is still needed (CAM gives O(1) already)
2. How to hold concepts during NSM/grammar decomposition into SPO
3. Whether the Luftschleuse airgap still makes sense
4. The relationship between SPO crystal, BindSpace, Blackboard, and LanceDB

## What Currently Exists (The Actual Code Path)

```
Text arrives
    │
    ▼
cortex.rs:278  →  deposit_evidence(input_fp, nars_tv)
    │
    ▼
awareness.rs:164  →  self.superposition = self.superposition.bind(&fp)  ← XOR BIND!
                      self.evidence_buffer.push((fp, tv))
    │
    ▼
awareness.rs:183  →  evaluate()  →  SD of confidence scores → GateState
    │
    ├── FLOW  → commit to DeltaLayer → commit_to(bind_space)
    ├── HOLD  → keep accumulating (superposition via XOR)
    └── BLOCK → discard, suggest style switch
```

**Problem 1:** `deposit_evidence` XOR-binds into ONE flat superposition.
The 3D SPO structure is destroyed. S, P, O planes are collapsed into one XOR soup.

**Problem 2:** The SPO Crystal (`spo.rs`) and SentenceCrystal (`sentence_crystal.rs`)
exist as SEPARATE modules. They're not wired into the awareness/cortex pipeline.
The cortex receives a flat fingerprint, not an SPO-decomposed triple.

**Problem 3:** The 90° orthogonal vector was for instant search. But:
- `cam_ops.rs` (159KB) provides O(1) content-addressable lookup
- `bind_space.rs` provides O(1) prefix:addr direct access
- `hdr_cascade.rs` provides progressive Hamming search
The 90° vector's job is already done three different ways.

**Problem 4:** The Luftschleuse was about airgapping write-through to prevent
XOR race conditions. But if we use bundle (not XOR) for write-back, and if
the awareness register should be PER-PLANE (not flat), the airgap boundary moves.

## What Should Exist

### The Real Pipeline

```
Text / Concept arrives
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: NSM/Grammar Decomposition                     │
│                                                          │
│  text → NSM primes → role parsing → agent/action/patient │
│       → grammar verb identification                      │
│       → S, P, O fingerprints (three separate vectors)    │
│                                                          │
│  Uses: grammar/nsm.rs, grammar/unified_parser.rs,        │
│        spo/deepnsm_integration.rs, spo/nsm_substrate.rs  │
│                                                          │
│  Output: SPOTriple { s_fp, p_fp, o_fp, verb_id, qualia } │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Three-Plane Awareness (replaces flat XOR)      │
│                                                          │
│  Three SEPARATE awareness registers:                     │
│                                                          │
│  awareness_s: int8[2048]  — subject plane                │
│  awareness_p: int8[2048]  — predicate plane              │
│  awareness_o: int8[2048]  — object plane                 │
│                                                          │
│  Each register soaks via bundle (saturating_add),         │
│  NOT XOR bind. Multiple concepts accumulate without       │
│  cancellation. int8 gives 256 levels = ~64-256 concepts   │
│  before saturation (the "forgiving" property).            │
│                                                          │
│  The three registers NEVER mix. S evidence only touches   │
│  the S register. This is what "3D non-dilution" means.   │
│                                                          │
│  CollapseGate evaluates EACH PLANE independently:         │
│  - S saturated + P saturated + O unsaturated = HOLD       │
│  - All three saturated = FLOW                             │
│  - Any plane contradictory = BLOCK for that plane         │
│                                                          │
│  The 8-term Faktorzerlegung runs on the TRIPLE of         │
│  registers, not on a flattened single fingerprint.        │
│                                                          │
│  Output: ThreeGateState { s_gate, p_gate, o_gate }       │
│          + interaction analysis (SP, PO, SO, SPO terms)   │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: CAM Index O(1) Lookup                          │
│                                                          │
│  OLD: 90° orthogonal vector → scan BindSpace              │
│  NEW: CAM fingerprint → addr in one hop                   │
│                                                          │
│  Three CAM lookups, one per plane:                        │
│    cam_s(s_fp) → addr_s     (subject address)             │
│    cam_p(p_fp) → addr_p     (predicate address)           │
│    cam_o(o_fp) → addr_o     (object address)              │
│                                                          │
│  The TRIPLE of addresses (addr_s, addr_p, addr_o) is      │
│  the 3D coordinate in the crystal. No hash needed —       │
│  the CAM gives you the address directly.                  │
│                                                          │
│  This replaces:                                           │
│  - 90° vector (instant search → CAM is already instant)   │
│  - grid_hash() in spo.rs (hash to 5×5×5 → CAM to addr)   │
│  - Linear scan of BindSpace (CAM is O(1))                 │
│                                                          │
│  The 5×5×5 crystal becomes a VIEW over BindSpace,          │
│  addressed by CAM, not a separate data structure.          │
└───────────┬─────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Masked Attention over LanceDB                  │
│                                                          │
│  With three CAM addresses, you can:                       │
│                                                          │
│  a) READ: Query LanceDB with a MASKED filter              │
│     - Fix S, vary P,O → "what does subject X do?"         │
│     - Fix P, vary S,O → "who does action Y?"              │
│     - Fix S,P, vary O → "what is X doing Y to?"           │
│                                                          │
│     The mask IS the partial query. No special encoding.    │
│     LanceDB column filters on the three addr columns.     │
│                                                          │
│  b) WRITE: Three-plane commit                             │
│     - Each plane writes to its own BindSpace range         │
│     - S → surface prefix 0x00-0x0F                        │
│     - P → fluid prefix 0x10-0x7F                          │
│     - O → node prefix 0x80-0xFF                           │
│                                                          │
│     The EDGE (the relationship) is stored as:              │
│     - LanceDB row: (addr_s, addr_p, addr_o, nars_tv,      │
│                      spo_fingerprint, metadata)            │
│     - The SPO fingerprint is S⊕Role_S⊕P⊕Role_P⊕O⊕Role_O │
│       (XOR bind is fine HERE because this is a READ-ONLY   │
│       encoding for similarity search, not mutable state)   │
│                                                          │
│  c) FOCUS: Attention mask from deepmsm                     │
│     - Learned per-plane attention weights                   │
│     - Which nibble positions in S matter for this P→O?     │
│     - This IS the "focus of attention" — not a separate     │
│       mechanism but a per-query weight on the CAM lookup    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## What Happens to the Luftschleuse?

The Luftschleuse airgap was between:
- READ world (immutable Arrow/Lance) and
- WRITE world (mutable BindSpace)

With three-plane awareness + CAM addressing, the boundary shifts:

```
OLD: concept → flat XOR → awareness → collapse → [LUFTSCHLEUSE] → BindSpace

NEW: concept → NSM decompose → three int8 registers
                                    │
                              ┌─────┴─────┐
                              │  Per-plane  │
                              │  collapse   │
                              └─────┬─────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                  S commit       P commit       O commit
                     │              │              │
                     └──────┬───────┘              │
                            │                      │
                      EDGE COMMIT ←────────────────┘
                            │
                    [LUFTSCHLEUSE]
                            │
                     ┌──────┴──────┐
                     │  LanceDB    │  (append-only, immutable after write)
                     │  row with   │  (addr_s, addr_p, addr_o, nars_tv,
                     │  3 addrs    │   spo_fp, metadata, timestamp)
                     └─────────────┘
```

The Luftschleuse moves DOWN. It's no longer between awareness and BindSpace.
It's between the three-plane commit and LanceDB. Because:

1. BindSpace writes are CHEAP (direct array index, no lock needed for single-writer)
2. The race condition concern was about multiple writers to the same addr
3. With CAM addressing, each concept gets a UNIQUE addr — no contention
4. The only shared mutable state that needs airgapping is the LanceDB append

So the Luftschleuse becomes the LanceDB transaction boundary:
- Bundle micro-deltas in BindSpace (fast, per-plane, no race)
- When batch is ready, atomically append to LanceDB (the airgap)
- LanceDB row is immutable after write (Arrow semantics)

## What Happens to the Blackboard?

The AwarenessBlackboard becomes the THREE-PLANE register:

```rust
pub struct ThreePlaneAwareness {
    /// Subject awareness — int8 soaking register
    s_register: [i8; REGISTER_SIZE],  // 2048 bytes = 16384 bits ÷ 8
    /// Predicate awareness
    p_register: [i8; REGISTER_SIZE],
    /// Object awareness  
    o_register: [i8; REGISTER_SIZE],
    
    /// Per-plane gate state
    s_gate: GateState,
    p_gate: GateState,
    o_gate: GateState,
    
    /// Evidence buffer (SPO triples, not flat fps)
    evidence: Vec<SPOTriple>,
    
    /// Cycle counter
    cycle: u64,
}

impl ThreePlaneAwareness {
    /// Deposit SPO-decomposed evidence
    pub fn deposit(&mut self, triple: &SPOTriple) {
        // Soak each plane independently via saturating_add
        for (i, byte) in triple.s_fp.as_bytes().iter().enumerate() {
            // Convert u8 bit pattern to signed contribution
            let contribution = (*byte as i8).wrapping_sub(128); // center at 0
            self.s_register[i] = self.s_register[i].saturating_add(
                (contribution as f32 * triple.confidence).round() as i8
            );
        }
        // Same for P and O registers...
        self.evidence.push(triple.clone());
    }
    
    /// Evaluate per-plane collapse gates
    pub fn evaluate(&mut self) -> ThreeGateState {
        self.s_gate = self.evaluate_plane(&self.s_register);
        self.p_gate = self.evaluate_plane(&self.p_register);
        self.o_gate = self.evaluate_plane(&self.o_register);
        
        ThreeGateState {
            s: self.s_gate,
            p: self.p_gate,
            o: self.o_gate,
            // Interaction: all FLOW = commit edge
            // Mixed: wait for lagging plane
            // Any BLOCK: contradiction in that plane
        }
    }
    
    fn evaluate_plane(&self, register: &[i8; REGISTER_SIZE]) -> GateState {
        // Saturation analysis: how many bytes are near ±127?
        let saturated = register.iter()
            .filter(|&&v| v.abs() > 100)
            .count();
        let total = REGISTER_SIZE;
        
        let saturation_ratio = saturated as f32 / total as f32;
        
        if saturation_ratio > 0.8 { GateState::Flow }      // deeply soaked
        else if saturation_ratio > 0.2 { GateState::Hold }  // still absorbing
        else { GateState::Block }                            // too dispersed
    }
}
```

The int8 register gives ~64-256 concepts before saturation because:
- Each `saturating_add` contributes ~1-4 to each byte
- int8 range is [-128, +127] = 256 levels
- At 1 unit per concept: 127 concepts before positive saturation
- At 2 units per concept (strong evidence): ~63 concepts
- This is the "forgiving" property — many concepts can soak in before the gate fires

## What Dies

| Old Concept | Why It Dies | Replacement |
|---|---|---|
| 90° orthogonal vector for instant search | CAM gives O(1) already | CAM index (cam_ops.rs) |
| Flat XOR superposition in awareness | Destroys 3D SPO non-dilution | Three int8 registers |
| grid_hash() for 5×5×5 crystal addressing | Loses information, collisions | CAM → addr directly |
| SPOCrystal as separate data structure | Duplicates BindSpace | BindSpace with three-column addr |
| Luftschleuse between awareness and BindSpace | BindSpace writes are single-writer | Moves to LanceDB boundary |
| SD-threshold collapse gate | No cumulative residual tracking | Per-plane saturation analysis |

## What Lives

| Concept | Why It Lives | Where |
|---|---|---|
| XOR for BIND (role binding) | Pure algebra, no mutation | BindEdge::bind(), encode_triple() |
| XOR for Hamming distance | Measurement, stateless | hamming_distance() |
| XOR for parity (error correction) | Single-writer, no race | ParityBlock::update_single() |
| Bundle (majority vote) for read-time merge | Read-only, no mutation | bind_space.bundle() |
| NSM decomposition pipeline | Already implemented, works | 6 files, 165KB |
| Satisfaction gate (Maslow hierarchy) | Orthogonal to plane structure | Per-LAYER, not per-plane |
| NARS revision rule | The physics, doesn't change | nars/inference.rs |
| 8-term Faktorzerlegung | The causal structure | Runs on the triple, not flat |

## What Changes About Faktorzerlegung

Currently: Faktorzerlegung runs on ONE fingerprint (the flattened XOR-bound triple).
The 8 terms are extracted by masking different bit ranges.

New: Faktorzerlegung runs on THREE registers (S, P, O).
The 8 terms come from comparing the three registers directly:

```
∅  = baseline (no planes)           — the null model
S  = subject main effect             — S register alone predicts outcome
P  = predicate main effect           — P register alone predicts outcome  
O  = object main effect              — O register alone predicts outcome
SP = subject × predicate interaction — S and P together predict more than S+P
PO = predicate × object interaction  — P and O together predict more than P+O
SO = subject × object interaction    — S and O together predict more than S+O
SPO = irreducible three-way          — the triple predicts more than all pairs combined
```

With three SEPARATE int8 registers, this is no longer bit-masking tricks on a flat vector.
It's genuine factorial analysis on three independent signals. The orthogonality assumption
that ChatGPT flagged as critical? It's ENFORCED by having separate registers that never mix.

## The Master Diagram

```
                        ┌──────────────┐
                        │  Text Input  │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │ NSM/Grammar Decompose│
                    │ (65 primes, 144 verbs)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    SPO Triple        │
                    │  {s_fp, p_fp, o_fp}  │
                    └──┬───────┬───────┬──┘
                       │       │       │
              ┌────────▼──┐ ┌─▼────┐ ┌▼────────┐
              │ S register│ │P reg │ │ O register│  ← int8 soaking
              │ (soak)    │ │(soak)│ │ (soak)    │     (bundle/sat_add)
              └────┬──────┘ └──┬───┘ └────┬─────┘
                   │           │          │
              ┌────▼──────┐ ┌──▼───┐ ┌───▼──────┐
              │ S gate    │ │P gate│ │ O gate    │  ← per-plane collapse
              │ (saturated│ │      │ │           │     (~64-256 concepts)
              │  = FLOW)  │ │      │ │           │
              └────┬──────┘ └──┬───┘ └────┬─────┘
                   │           │          │
              ┌────▼───────────▼──────────▼─────┐
              │         8-term Faktorzerlegung    │  ← genuine factorial
              │    (S, P, O registers are inputs)  │     (not bit-mask tricks)
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         CAM Addressing              │
              │  cam(s_fp)→addr_s                   │  ← O(1) per plane
              │  cam(p_fp)→addr_p                   │
              │  cam(o_fp)→addr_o                   │
              └────────────────┬──────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │         BindSpace Write             │
              │  write_at(addr_s, s_register)       │  ← direct array index
              │  write_at(addr_p, p_register)       │     per plane, no contention
              │  write_at(addr_o, o_register)       │
              └────────────────┬──────────────────┘
                               │
                      [LUFTSCHLEUSE]                   ← the only airgap left
                               │
              ┌────────────────▼──────────────────┐
              │         LanceDB Append              │
              │  row: (addr_s, addr_p, addr_o,      │  ← immutable after write
              │        nars_tv, spo_fp, metadata)    │
              └───────────────────────────────────┘
                               │
              ┌────────────────▼──────────────────┐
              │    Masked Query (focus of attention) │
              │  Fix S → what does this subject do?  │  ← LanceDB column filter
              │  Fix P → who does this action?        │     with deepmsm attention
              │  Fix S,P → what is X doing Y to?      │     weights per nibble
              └───────────────────────────────────┘
```

## Impact on the Two Punktlandung Sessions

### deepmsm session gains:
- Attention masks now operate on THREE planes, not one flat vector
- TransitionMatrix tracks per-plane transitions (S→S', P→P', O→O')
- VAMPE calibrates three independent σ-band spectra
- CK test validates Markov property per plane

### jaxpi session gains:
- Term balancing is now NATIVE — three separate registers = three separate gradient norms
- Residual monitor tracks three convergence curves, one per plane
- Causal weighting applies per-plane (S evidence gates S revision, not global gate)

### Both sessions get simpler:
- No need to unmask bit ranges to recover plane contributions
- The Faktorzerlegung is just comparing three arrays, not doing algebra on one
- The orthogonality ChatGPT worried about is ENFORCED by register separation

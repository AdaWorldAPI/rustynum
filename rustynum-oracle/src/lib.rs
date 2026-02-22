//! # rustynum-oracle
//!
//! Three-temperature holographic oracle with exhaustive capacity sweep.
//!
//! This crate implements two things:
//!
//! 1. **Capacity sweep** — test every combination of D, base, signing, axes, K,
//!    and bind depth to find the sweet spot for holographic storage.
//!
//! 2. **Three-temperature oracle** — one oracle per entity with hot/warm/cold tiers,
//!    overexposure-triggered flush, and coefficient-as-canonical-storage.
//!
//! The oracle is the thinking/storage tier. The 8KB CogRecordV3 in `rustynum-holo`
//! remains as the fast-rejection search tier.

pub mod linalg;
pub mod sweep;
pub mod oracle;
pub mod organic;
pub mod aiwar_ghost;
pub mod ghost_discovery;
pub mod recognize;
pub mod nars;

pub use linalg::{
    cholesky_solve, condition_number,
    upsample_to_f32, downsample_to_base,
};

pub use sweep::{
    Base, DIMS, BASES, AXES, BUNDLE_SIZES, BIND_DEPTHS,
    generate_template, generate_templates,
    bind, bind_deep, bundle,
    measure_recovery, measure_recovery_multiaxis,
    measure_bell_coefficient,
    run_sweep,
    RecoveryResult, AxisResult, MultiAxisResult,
};

pub use oracle::{
    Oracle, Temperature, FlushAction,
    TemplateLibrary,
    MaterializedHolograph,
};

pub use organic::{
    XTransPattern, MultiResPattern,
    receptivity, organic_write, organic_write_f32, organic_read,
    OrganicWAL, WriteResult,
    PlasticityTracker, AbsorptionTracker,
    FlushResult, organic_flush,
    OrganicResult, measure_recovery_organic, run_organic_sweep,
    organic_results_to_csv,
    StressPoint, stress_test_absorption,
    run_absorption_stress_test, run_plasticity_comparison,
    ORGANIC_CHANNELS, ORGANIC_PLASTICITY,
};
pub use organic::FlushAction as OrganicFlushAction;

pub use nars::{
    unbind, forward_bind, reverse_unbind,
    Entity as NarsEntity, Role as NarsRole,
    reverse_trace, CausalTrace, TraceStep,
    granger_signal, granger_scan,
    find_similar_pairs, SimilarPair,
};

pub use recognize::{
    Projector64K, Recognizer, RecognitionResult, RecognitionMethod,
    ExperimentResult,
    hamming_64k, hamming_similarity_64k,
    run_recognition_experiment, run_recognition_sweep,
    print_recognition_results, run_recognition,
};

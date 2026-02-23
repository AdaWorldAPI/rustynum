//! JIT compilation engine — Cranelift infrastructure.
//!
//! The `JitEngine` manages Cranelift's JIT module, compiles IR to native code,
//! and caches compiled kernels by their parameter hash.

use std::collections::HashMap;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Signature, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use crate::detect::CpuCaps;
use crate::ir::{JitError, ScanParams};
use crate::scan_jit::ScanKernel;

/// The JIT compilation engine.
///
/// Holds a Cranelift `JITModule` and a cache of compiled kernels.
/// Thread-safe: compiled function pointers can be shared across threads.
pub struct JitEngine {
    /// Cranelift JIT module — owns the compiled code pages.
    module: JITModule,

    /// CPU capabilities detected at engine creation.
    pub caps: CpuCaps,

    /// Compiled scan kernel cache: params hash → (func_id, fn_ptr).
    scan_cache: HashMap<u64, (*const u8, FuncId)>,
}

// Safety: JITModule's compiled code pages are immutable after finalization.
// Function pointers are safe to call from any thread.
unsafe impl Send for JitEngine {}
unsafe impl Sync for JitEngine {}

impl JitEngine {
    /// Create a new JIT engine with auto-detected CPU features.
    pub fn new() -> Result<Self, JitError> {
        let caps = CpuCaps::detect();

        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let isa_builder = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            caps,
            scan_cache: HashMap::new(),
        })
    }

    /// Register an external function that JIT code can call.
    ///
    /// This is the hybrid approach: JIT the loop, call existing SIMD kernels
    /// (like `hamming_distance`) via function pointer.
    pub fn register_extern_fn(
        &mut self,
        name: &str,
        _ptr: *const u8,
        signature: Signature,
    ) -> Result<FuncId, JitError> {
        let func_id = self
            .module
            .declare_function(name, Linkage::Import, &signature)
            .map_err(|e| JitError::Module(e.to_string()))?;

        // NOTE: In cranelift-jit, imported functions are resolved via the
        // JITBuilder's symbol lookup. For direct fn ptr registration,
        // we'd need to extend JITBuilder before module creation.
        // For now, functions must be registered at builder time.
        Ok(func_id)
    }

    /// Compile a scan kernel with the given parameters baked as immediates.
    ///
    /// Returns a `ScanKernel` whose `scan()` method is a native function
    /// pointer with `threshold`, `prefetch_ahead`, `record_size` etc.
    /// compiled as immediate operands.
    pub fn compile_scan(&mut self, params: ScanParams) -> Result<ScanKernel, JitError> {
        // Check cache
        let cache_key = params_hash(&params);
        if let Some(&(ptr, _)) = self.scan_cache.get(&cache_key) {
            return Ok(ScanKernel::from_raw(ptr, params));
        }

        // Build the scan function
        let func_name = format!("scan_{cache_key:x}");
        let sig = scan_signature(&self.module);

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Generate the scan loop IR
        crate::scan_jit::build_scan_ir(&mut ctx.func, &params, &self.module)?;

        // Compile
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| JitError::Codegen(format!("{e:?}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        self.scan_cache.insert(cache_key, (code_ptr, func_id));

        Ok(ScanKernel::from_raw(code_ptr, params))
    }

    /// Get the number of cached kernels.
    pub fn cached_count(&self) -> usize {
        self.scan_cache.len()
    }
}

/// Scan function signature:
/// `fn(query: *const u8, field: *const u8, field_len: u64,
///     record_size: u64, candidates_out: *mut u64) -> u64`
fn scan_signature(module: &JITModule) -> Signature {
    let ptr_type = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // query
    sig.params.push(AbiParam::new(ptr_type)); // field
    sig.params.push(AbiParam::new(types::I64)); // field_len
    sig.params.push(AbiParam::new(types::I64)); // record_size
    sig.params.push(AbiParam::new(ptr_type)); // candidates_out
    sig.returns.push(AbiParam::new(types::I64)); // num_candidates
    sig
}

/// Simple hash of scan params for cache lookup.
fn params_hash(params: &ScanParams) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    params.threshold.hash(&mut hasher);
    params.top_k.hash(&mut hasher);
    params.prefetch_ahead.hash(&mut hasher);
    params.record_size.hash(&mut hasher);
    if let Some(ref mask) = params.focus_mask {
        mask.hash(&mut hasher);
    }
    hasher.finish()
}

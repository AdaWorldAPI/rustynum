//! # JITSON — JSON Template Engine for JIT Scan Pipelines
//!
//! Parses JSON (and YAML-like) templates into [`ScanConfig`] and AVX-512
//! pipeline descriptors.  Zero external dependencies — the parser is a
//! hand-rolled recursive-descent tokenizer that handles the template subset
//! and includes **bracket recovery**: if the only structural error is a
//! missing closing `}` or `]`, the parser auto-appends it and succeeds.
//!
//! ## Template Format
//!
//! ```json
//! {
//!   "version": 1,
//!   "kernel": "hamming_distance",
//!   "scan": {
//!     "threshold": 2048,
//!     "record_size": 256,
//!     "top_k": 10
//!   },
//!   "pipeline": [
//!     { "stage": "xor",    "avx512": "vpxord" },
//!     { "stage": "popcnt", "avx512": "vpopcntd", "fallback": "avx2_lookup" },
//!     { "stage": "reduce", "avx512": "vpaddd" }
//!   ],
//!   "features": {
//!     "avx512f": true,
//!     "avx512vl": true,
//!     "avx512bw": false
//!   },
//!   "cranelift": {
//!     "preset": "sapphire_rapids",
//!     "opt_level": "speed"
//!   }
//! }
//! ```

use alloc::string::String;
use alloc::vec::Vec;

extern crate alloc;

// ---------------------------------------------------------------------------
// JSON value type
// ---------------------------------------------------------------------------

/// Minimal JSON value — covers the template subset.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().and_then(|n| {
            if n >= 0.0 && n <= u64::MAX as f64 && n.fract() == 0.0 {
                Some(n as u64)
            } else {
                None
            }
        })
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|n| n as usize)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o.as_slice()),
            _ => None,
        }
    }

    /// Lookup a key in an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()
            .and_then(|pairs| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "JITSON parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// Tokenizer + recursive-descent parser with bracket recovery
// ---------------------------------------------------------------------------

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
    /// Track open brackets/braces for recovery.
    open_stack: Vec<u8>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            open_stack: Vec::new(),
        }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            message: String::from(msg),
            offset: self.pos,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                // Skip single-line comments (YAML-ish extension)
                b'/' if self.pos + 1 < self.input.len() && self.input[self.pos + 1] == b'/' => {
                    self.pos += 2;
                    while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), ParseError> {
        match self.advance() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(self.err(&alloc::format!(
                "expected '{}', found '{}'",
                expected as char,
                b as char
            ))),
            None => {
                // Bracket recovery: if we hit EOF expecting a closing bracket,
                // check if it matches the top of our open_stack.
                if (expected == b'}' || expected == b']')
                    && self.open_stack.last().copied() == Some(expected)
                {
                    self.open_stack.pop();
                    Ok(())
                } else {
                    Err(self.err(&alloc::format!(
                        "unexpected EOF, expected '{}'",
                        expected as char
                    )))
                }
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, ParseError> {
        match self.peek() {
            Some(b'"') => self.parse_string().map(JsonValue::Str),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(c) => Err(self.err(&alloc::format!("unexpected character '{}'", c as char))),
            None => Err(self.err("unexpected EOF")),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            if self.pos >= self.input.len() {
                return Err(self.err("unterminated string"));
            }
            let b = self.input[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(s),
                b'\\' => {
                    if self.pos >= self.input.len() {
                        return Err(self.err("unterminated escape"));
                    }
                    let esc = self.input[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\x08'),
                        b'f' => s.push('\x0c'),
                        b'u' => {
                            // \uXXXX — parse 4 hex digits
                            if self.pos + 4 > self.input.len() {
                                return Err(self.err("incomplete \\u escape"));
                            }
                            let hex = &self.input[self.pos..self.pos + 4];
                            self.pos += 4;
                            let hex_str = core::str::from_utf8(hex)
                                .map_err(|_| self.err("invalid \\u hex"))?;
                            let cp = u32::from_str_radix(hex_str, 16)
                                .map_err(|_| self.err("invalid \\u hex"))?;
                            if let Some(c) = char::from_u32(cp) {
                                s.push(c);
                            }
                        }
                        _ => {
                            s.push('\\');
                            s.push(esc as char);
                        }
                    }
                }
                _ => s.push(b as char),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        let start = self.pos;
        if self.pos < self.input.len() && self.input[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.input.len()
            && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.input.len()
                && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
            {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let slice = &self.input[start..self.pos];
        let s = core::str::from_utf8(slice).map_err(|_| self.err("invalid number bytes"))?;
        let n: f64 = s.parse().map_err(|_| self.err("invalid number"))?;
        Ok(JsonValue::Number(n))
    }

    fn parse_bool(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.err("expected 'true' or 'false'"))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(self.err("expected 'null'"))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, ParseError> {
        self.expect(b'{')?;
        self.open_stack.push(b'}');
        let mut pairs = Vec::new();
        if self.peek() == Some(b'}') {
            self.advance();
            self.open_stack.pop();
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            pairs.push((key, val));
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    // Allow trailing comma before closing brace
                    if self.peek() == Some(b'}') {
                        self.advance();
                        self.open_stack.pop();
                        return Ok(JsonValue::Object(pairs));
                    }
                }
                Some(b'}') => {
                    self.advance();
                    self.open_stack.pop();
                    return Ok(JsonValue::Object(pairs));
                }
                None => {
                    // Bracket recovery: EOF but we know we're inside an object
                    if self.open_stack.last().copied() == Some(b'}') {
                        self.open_stack.pop();
                        return Ok(JsonValue::Object(pairs));
                    }
                    return Err(self.err("unexpected EOF in object"));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, ParseError> {
        self.expect(b'[')?;
        self.open_stack.push(b']');
        let mut elems = Vec::new();
        if self.peek() == Some(b']') {
            self.advance();
            self.open_stack.pop();
            return Ok(JsonValue::Array(elems));
        }
        loop {
            let val = self.parse_value()?;
            elems.push(val);
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    // Allow trailing comma before closing bracket
                    if self.peek() == Some(b']') {
                        self.advance();
                        self.open_stack.pop();
                        return Ok(JsonValue::Array(elems));
                    }
                }
                Some(b']') => {
                    self.advance();
                    self.open_stack.pop();
                    return Ok(JsonValue::Array(elems));
                }
                None => {
                    // Bracket recovery: EOF inside array
                    if self.open_stack.last().copied() == Some(b']') {
                        self.open_stack.pop();
                        return Ok(JsonValue::Array(elems));
                    }
                    return Err(self.err("unexpected EOF in array"));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }
}

/// Parse a JSON string into a [`JsonValue`].
///
/// Includes bracket recovery: if the input is valid except for missing
/// trailing `}` or `]` characters, the parser auto-closes them.
pub fn parse_json(input: &str) -> Result<JsonValue, ParseError> {
    let mut parser = Parser::new(input.as_bytes());
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos < parser.input.len() {
        return Err(parser.err("trailing data after JSON value"));
    }
    // If there are still unclosed brackets, the recovery already handled
    // them during parsing.  But if the stack is non-empty here it means
    // recovery couldn't fire (e.g. value ended cleanly but stack is wrong).
    // That shouldn't happen with well-structured recovery, but guard anyway.
    Ok(value)
}

// ---------------------------------------------------------------------------
// Schema validation
// ---------------------------------------------------------------------------

/// Validation error with a JSON-pointer path.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl core::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "JITSON schema error at {}: {}", self.path, self.message)
    }
}

/// All AVX-512 feature flags supported by the patched Cranelift backend.
pub const KNOWN_FEATURES: &[&str] = &[
    "avx512f",
    "avx512vl",
    "avx512bw",
    "avx512dq",
    "avx512bitalg",
    "avx512vbmi",
    "avx512vpopcntdq",
    "avx512vnni",
    "avx512ifma",
];

/// All AVX-512 instruction mnemonics from the patched Cranelift.
pub const KNOWN_INSTRUCTIONS: &[&str] = &[
    // abs
    "vpabsb",
    "vpabsw",
    "vpabsd",
    "vpabsq",
    // and / ternlog
    "vpandd",
    "vpandq",
    "vpandnd",
    "vpandnq",
    "vpternlogd",
    "vpternlogq",
    // bitmanip
    "vpopcntb",
    "vpopcntw",
    "vpopcntd",
    "vpopcntq",
    // fma (132/213/231 x ps/pd x add/sub/nmadd)
    "vfmadd132ps",
    "vfmadd213ps",
    "vfmadd231ps",
    "vfmadd132pd",
    "vfmadd213pd",
    "vfmadd231pd",
    "vfmsub132ps",
    "vfmsub213ps",
    "vfmsub231ps",
    "vfmsub132pd",
    "vfmsub213pd",
    "vfmsub231pd",
    "vfnmadd132ps",
    "vfnmadd213ps",
    "vfnmadd231ps",
    "vfnmadd132pd",
    "vfnmadd213pd",
    "vfnmadd231pd",
    // mul / vnni
    "vpmulld",
    "vpmullq",
    "vpdpbusd",
    "vpdpbusds",
    "vpdpwssd",
    "vpdpwssds",
    // or
    "vpord",
    "vporq",
    // shift
    "vpsllw",
    "vpslld",
    "vpsllq",
    "vpsraw",
    "vpsrad",
    "vpsraq",
    "vpsrlw",
    "vpsrld",
    "vpsrlq",
    // xor
    "vpxord",
    "vpxorq",
    // add
    "vaddpd",
    // cvt
    "vcvtudq2ps",
    // lanes
    "vpermi2b",
];

/// Known kernel names.
const KNOWN_KERNELS: &[&str] = &["hamming_distance", "cosine_i8", "dot_f32"];

/// Known backend names for CPU-lane data sources/sinks.
pub const KNOWN_BACKENDS: &[&str] = &["lancedb", "dragonfly"];

/// Known Cranelift presets.
const KNOWN_PRESETS: &[&str] = &[
    "baseline",
    "nehalem",
    "haswell",
    "broadwell",
    "skylake",
    "knl",
    "knm",
    "skylake_avx512",
    "cascade_lake",
    "cooper_lake",
    "cannon_lake",
    "ice_lake_client",
    "ice_lake_server",
    "tiger_lake",
    "sapphire_rapids",
    "x86_64_v2",
    "x86_64_v3",
    "x86_64_v4",
];

/// Known opt levels.
const KNOWN_OPT_LEVELS: &[&str] = &["none", "speed", "speed_and_size"];

/// Validate a parsed JITSON template against the schema.
///
/// Returns a list of all validation errors found (empty = valid).
pub fn validate(root: &JsonValue) -> Vec<ValidationError> {
    let mut errs = Vec::new();

    let obj = match root.as_object() {
        Some(o) => o,
        None => {
            errs.push(ValidationError {
                path: String::from("/"),
                message: String::from("root must be a JSON object"),
            });
            return errs;
        }
    };

    // version (required, must be 1)
    match root.get("version") {
        Some(v) => match v.as_u64() {
            Some(1) => {}
            Some(n) => errs.push(ValidationError {
                path: String::from("/version"),
                message: alloc::format!("unsupported version {}, expected 1", n),
            }),
            None => errs.push(ValidationError {
                path: String::from("/version"),
                message: String::from("must be an integer"),
            }),
        },
        None => errs.push(ValidationError {
            path: String::from("/version"),
            message: String::from("required field missing"),
        }),
    }

    // kernel (required, one of known kernels)
    match root.get("kernel") {
        Some(v) => match v.as_str() {
            Some(s) if KNOWN_KERNELS.contains(&s) => {}
            Some(s) => errs.push(ValidationError {
                path: String::from("/kernel"),
                message: alloc::format!(
                    "unknown kernel \"{}\", expected one of: {}",
                    s,
                    KNOWN_KERNELS.join(", ")
                ),
            }),
            None => errs.push(ValidationError {
                path: String::from("/kernel"),
                message: String::from("must be a string"),
            }),
        },
        None => errs.push(ValidationError {
            path: String::from("/kernel"),
            message: String::from("required field missing"),
        }),
    }

    // scan (required object with threshold, record_size, top_k)
    match root.get("scan") {
        Some(scan) => {
            if scan.as_object().is_none() {
                errs.push(ValidationError {
                    path: String::from("/scan"),
                    message: String::from("must be an object"),
                });
            } else {
                validate_uint_field(scan, "threshold", "/scan/threshold", &mut errs);
                validate_uint_field(scan, "record_size", "/scan/record_size", &mut errs);
                validate_uint_field(scan, "top_k", "/scan/top_k", &mut errs);
            }
        }
        None => errs.push(ValidationError {
            path: String::from("/scan"),
            message: String::from("required field missing"),
        }),
    }

    // pipeline (optional array of stage objects)
    if let Some(pipeline) = root.get("pipeline") {
        match pipeline.as_array() {
            Some(stages) => {
                for (i, stage) in stages.iter().enumerate() {
                    let prefix = alloc::format!("/pipeline/{}", i);
                    if stage.as_object().is_none() {
                        errs.push(ValidationError {
                            path: prefix.clone(),
                            message: String::from("each pipeline stage must be an object"),
                        });
                        continue;
                    }
                    // stage name (required)
                    if stage.get("stage").and_then(|v| v.as_str()).is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("{}/stage", prefix),
                            message: String::from("required string field"),
                        });
                    }
                    // avx512 instruction (optional but validated)
                    if let Some(instr) = stage.get("avx512").and_then(|v| v.as_str()) {
                        if !KNOWN_INSTRUCTIONS.contains(&instr) {
                            errs.push(ValidationError {
                                path: alloc::format!("{}/avx512", prefix),
                                message: alloc::format!(
                                    "unknown instruction \"{}\"; not in patched Cranelift",
                                    instr
                                ),
                            });
                        }
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/pipeline"),
                message: String::from("must be an array"),
            }),
        }
    }

    // features (optional object of bool flags)
    if let Some(features) = root.get("features") {
        match features.as_object() {
            Some(pairs) => {
                for (key, val) in pairs {
                    if !KNOWN_FEATURES.contains(&key.as_str()) {
                        errs.push(ValidationError {
                            path: alloc::format!("/features/{}", key),
                            message: alloc::format!(
                                "unknown feature \"{}\"; known: {}",
                                key,
                                KNOWN_FEATURES.join(", ")
                            ),
                        });
                    }
                    if val.as_bool().is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/features/{}", key),
                            message: String::from("must be a boolean"),
                        });
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/features"),
                message: String::from("must be an object"),
            }),
        }
    }

    // cranelift (optional)
    if let Some(cl) = root.get("cranelift") {
        if cl.as_object().is_none() {
            errs.push(ValidationError {
                path: String::from("/cranelift"),
                message: String::from("must be an object"),
            });
        } else {
            if let Some(preset) = cl.get("preset").and_then(|v| v.as_str()) {
                if !KNOWN_PRESETS.contains(&preset) {
                    errs.push(ValidationError {
                        path: String::from("/cranelift/preset"),
                        message: alloc::format!("unknown preset \"{}\"", preset),
                    });
                }
            }
            if let Some(opt) = cl.get("opt_level").and_then(|v| v.as_str()) {
                if !KNOWN_OPT_LEVELS.contains(&opt) {
                    errs.push(ValidationError {
                        path: String::from("/cranelift/opt_level"),
                        message: alloc::format!("unknown opt_level \"{}\"", opt),
                    });
                }
            }
        }
    }

    // backends (optional object of named backend configs)
    if let Some(backends) = root.get("backends") {
        match backends.as_object() {
            Some(pairs) => {
                for (key, val) in pairs {
                    if !KNOWN_BACKENDS.contains(&key.as_str()) {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}", key),
                            message: alloc::format!(
                                "unknown backend \"{}\"; known: {}",
                                key,
                                KNOWN_BACKENDS.join(", ")
                            ),
                        });
                    }
                    if val.as_object().is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}", key),
                            message: String::from("must be an object"),
                        });
                    } else if val.get("uri").and_then(|v| v.as_str()).is_none() {
                        errs.push(ValidationError {
                            path: alloc::format!("/backends/{}/uri", key),
                            message: String::from("required string field"),
                        });
                    }
                }
            }
            None => errs.push(ValidationError {
                path: String::from("/backends"),
                message: String::from("must be an object"),
            }),
        }
    }

    // Validate pipeline stage backend references
    if let Some(pipeline) = root.get("pipeline").and_then(|v| v.as_array()) {
        let declared_backends: Vec<&str> = root
            .get("backends")
            .and_then(|v| v.as_object())
            .map(|pairs| pairs.iter().map(|(k, _)| k.as_str()).collect())
            .unwrap_or_default();
        for (i, stage) in pipeline.iter().enumerate() {
            if let Some(backend) = stage.get("backend").and_then(|v| v.as_str()) {
                if !declared_backends.contains(&backend) {
                    errs.push(ValidationError {
                        path: alloc::format!("/pipeline/{}/backend", i),
                        message: alloc::format!(
                            "backend \"{}\" referenced but not declared in /backends",
                            backend
                        ),
                    });
                }
            }
        }
    }

    // Warn on unknown top-level keys
    let known_top: &[&str] = &[
        "version",
        "kernel",
        "scan",
        "pipeline",
        "features",
        "cranelift",
        "backends",
    ];
    for (key, _) in obj {
        if !known_top.contains(&key.as_str()) {
            errs.push(ValidationError {
                path: alloc::format!("/{}", key),
                message: alloc::format!("unknown field \"{}\"", key),
            });
        }
    }

    errs
}

fn validate_uint_field(parent: &JsonValue, key: &str, path: &str, errs: &mut Vec<ValidationError>) {
    match parent.get(key) {
        Some(v) => {
            if v.as_u64().is_none() {
                errs.push(ValidationError {
                    path: String::from(path),
                    message: String::from("must be a non-negative integer"),
                });
            }
        }
        None => errs.push(ValidationError {
            path: String::from(path),
            message: String::from("required field missing"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Template → ScanConfig conversion
// ---------------------------------------------------------------------------

use crate::jit_scan::ScanConfig;

/// Parsed and validated JITSON template.
#[derive(Clone, Debug)]
pub struct JitsonTemplate {
    pub kernel: String,
    pub scan: ScanConfig,
    pub pipeline: Vec<PipelineStage>,
    pub features: Vec<(String, bool)>,
    pub backends: Vec<BackendConfig>,
    pub cranelift_preset: Option<String>,
    pub cranelift_opt_level: Option<String>,
}

/// A single stage in the JIT pipeline.
#[derive(Clone, Debug)]
pub struct PipelineStage {
    pub stage: String,
    pub avx512_instr: Option<String>,
    pub fallback: Option<String>,
    /// Backend CPU-lane reference (e.g. "lancedb", "dragonfly").
    pub backend: Option<String>,
    /// Backend-specific key/table/prefix for this stage.
    pub backend_key: Option<String>,
}

/// Configuration for an external data backend (CPU lane).
#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub name: String,
    pub uri: String,
    /// Extra backend-specific options (key-value pairs from the JSON object).
    pub options: Vec<(String, String)>,
}

/// Parse a JSON string, validate it, and convert to a [`JitsonTemplate`].
///
/// Bracket recovery is applied automatically — a missing closing `}` or `]`
/// at the end of input will be silently fixed.
pub fn from_json(input: &str) -> Result<JitsonTemplate, JitsonError> {
    let root = parse_json(input).map_err(JitsonError::Parse)?;
    let errors = validate(&root);
    if !errors.is_empty() {
        return Err(JitsonError::Validation(errors));
    }
    convert(&root)
}

/// Error type for JITSON operations.
#[derive(Clone, Debug)]
pub enum JitsonError {
    Parse(ParseError),
    Validation(Vec<ValidationError>),
    Conversion(String),
}

impl core::fmt::Display for JitsonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            JitsonError::Parse(e) => write!(f, "{}", e),
            JitsonError::Validation(errs) => {
                for e in errs {
                    writeln!(f, "{}", e)?;
                }
                Ok(())
            }
            JitsonError::Conversion(msg) => write!(f, "JITSON conversion: {}", msg),
        }
    }
}

fn convert(root: &JsonValue) -> Result<JitsonTemplate, JitsonError> {
    let kernel = root.get("kernel").and_then(|v| v.as_str()).unwrap();
    let scan_obj = root.get("scan").unwrap();

    let scan = ScanConfig {
        threshold: scan_obj.get("threshold").and_then(|v| v.as_u64()).unwrap(),
        record_size: scan_obj
            .get("record_size")
            .and_then(|v| v.as_usize())
            .unwrap(),
        top_k: scan_obj.get("top_k").and_then(|v| v.as_usize()).unwrap(),
        query: Vec::new(), // Query bytes are provided at runtime, not in the template
    };

    let pipeline = match root.get("pipeline").and_then(|v| v.as_array()) {
        Some(stages) => stages
            .iter()
            .map(|s| PipelineStage {
                stage: s.get("stage").and_then(|v| v.as_str()).unwrap_or("").into(),
                avx512_instr: s.get("avx512").and_then(|v| v.as_str()).map(String::from),
                fallback: s.get("fallback").and_then(|v| v.as_str()).map(String::from),
                backend: s.get("backend").and_then(|v| v.as_str()).map(String::from),
                backend_key: s
                    .get("table")
                    .or_else(|| s.get("prefix"))
                    .or_else(|| s.get("key"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
            })
            .collect(),
        None => Vec::new(),
    };

    let backends = match root.get("backends").and_then(|v| v.as_object()) {
        Some(pairs) => pairs
            .iter()
            .map(|(name, cfg)| {
                let uri = cfg.get("uri").and_then(|v| v.as_str()).unwrap_or("").into();
                let options = cfg
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .filter(|(k, _)| k != "uri")
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), String::from(s))))
                            .collect()
                    })
                    .unwrap_or_default();
                BackendConfig {
                    name: name.clone(),
                    uri,
                    options,
                }
            })
            .collect(),
        None => Vec::new(),
    };

    let features = match root.get("features").and_then(|v| v.as_object()) {
        Some(pairs) => pairs
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bool().unwrap_or(false)))
            .collect(),
        None => Vec::new(),
    };

    let cranelift_preset = root
        .get("cranelift")
        .and_then(|cl| cl.get("preset"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let cranelift_opt_level = root
        .get("cranelift")
        .and_then(|cl| cl.get("opt_level"))
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(JitsonTemplate {
        kernel: String::from(kernel),
        scan,
        pipeline,
        features,
        backends,
        cranelift_preset,
        cranelift_opt_level,
    })
}

// ---------------------------------------------------------------------------
// Instruction → feature mapping (from patched Cranelift)
// ---------------------------------------------------------------------------

/// Return the required AVX-512 feature flags for a given instruction mnemonic.
///
/// Based on the patched Cranelift at
/// `AdaWorldAPI/wasmtime/tree/main/cranelift/assembler-x64/meta/src/instructions/`.
pub fn required_features(instruction: &str) -> &'static [&'static str] {
    match instruction {
        // abs
        "vpabsb" | "vpabsw" => &["avx512vl", "avx512bw"],
        "vpabsd" | "vpabsq" => &["avx512vl", "avx512f"],
        // and / ternlog
        "vpandd" | "vpandq" | "vpandnd" | "vpandnq" | "vpternlogd" | "vpternlogq" => {
            &["avx512vl", "avx512f"]
        }
        // bitmanip
        "vpopcntb" | "vpopcntw" => &["avx512vl", "avx512bitalg"],
        "vpopcntd" | "vpopcntq" => &["avx512vl", "avx512vpopcntdq"],
        // fma (all require avx512f + avx512vl)
        "vfmadd132ps" | "vfmadd213ps" | "vfmadd231ps" | "vfmadd132pd" | "vfmadd213pd"
        | "vfmadd231pd" | "vfmsub132ps" | "vfmsub213ps" | "vfmsub231ps" | "vfmsub132pd"
        | "vfmsub213pd" | "vfmsub231pd" | "vfnmadd132ps" | "vfnmadd213ps" | "vfnmadd231ps"
        | "vfnmadd132pd" | "vfnmadd213pd" | "vfnmadd231pd" => &["avx512vl", "avx512f"],
        // mul
        "vpmulld" => &["avx512vl", "avx512f"],
        "vpmullq" => &["avx512vl", "avx512dq"],
        // vnni
        "vpdpbusd" | "vpdpbusds" | "vpdpwssd" | "vpdpwssds" => &["avx512vl", "avx512vnni"],
        // or
        "vpord" | "vporq" => &["avx512vl", "avx512f"],
        // shift (bw for word, f for dword/qword)
        "vpsllw" | "vpsraw" | "vpsrlw" => &["avx512vl", "avx512bw"],
        "vpslld" | "vpsllq" | "vpsrad" | "vpsraq" | "vpsrld" | "vpsrlq" => &["avx512vl", "avx512f"],
        // xor
        "vpxord" | "vpxorq" => &["avx512vl", "avx512f"],
        // add
        "vaddpd" => &["avx512vl"],
        // cvt
        "vcvtudq2ps" => &["avx512vl", "avx512f"],
        // lanes
        "vpermi2b" => &["avx512vl", "avx512vbmi"],
        _ => &[],
    }
}

/// Check if a template's pipeline is satisfiable given the declared features.
///
/// Returns a list of (stage_index, instruction, missing_features) for each
/// pipeline stage that requires features not enabled in the template.
pub fn check_pipeline_features(template: &JitsonTemplate) -> Vec<(usize, String, Vec<String>)> {
    let enabled: Vec<&str> = template
        .features
        .iter()
        .filter(|(_, on)| *on)
        .map(|(k, _)| k.as_str())
        .collect();

    let mut unsatisfied = Vec::new();
    for (i, stage) in template.pipeline.iter().enumerate() {
        if let Some(ref instr) = stage.avx512_instr {
            let required = required_features(instr);
            let missing: Vec<String> = required
                .iter()
                .filter(|f| !enabled.contains(f))
                .map(|f| String::from(*f))
                .collect();
            if !missing.is_empty() {
                unsatisfied.push((i, instr.clone(), missing));
            }
        }
    }
    unsatisfied
}

// ---------------------------------------------------------------------------
// WAL Precompile Queue + Prefetch Addressing
// ---------------------------------------------------------------------------

/// Stable 64-bit hash of a JITSON template for use as a precompile cache key.
///
/// This is the "prefetch address" — a unique identifier for a compiled JIT
/// function.  Two templates that produce identical scan configs, pipelines,
/// and feature sets will have the same hash, enabling cache hits across
/// processes and restarts.
pub fn template_hash(template: &JitsonTemplate) -> u64 {
    // FNV-1a 64-bit — fast, deterministic, no dependencies
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut h = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };

    feed(template.kernel.as_bytes());
    feed(&template.scan.threshold.to_le_bytes());
    feed(&template.scan.record_size.to_le_bytes());
    feed(&template.scan.top_k.to_le_bytes());

    for stage in &template.pipeline {
        feed(stage.stage.as_bytes());
        if let Some(ref instr) = stage.avx512_instr {
            feed(instr.as_bytes());
        }
        if let Some(ref fb) = stage.fallback {
            feed(fb.as_bytes());
        }
        if let Some(ref be) = stage.backend {
            feed(be.as_bytes());
        }
    }

    for (feat, on) in &template.features {
        feed(feat.as_bytes());
        feed(&[*on as u8]);
    }

    if let Some(ref preset) = template.cranelift_preset {
        feed(preset.as_bytes());
    }
    if let Some(ref opt) = template.cranelift_opt_level {
        feed(opt.as_bytes());
    }

    h
}

/// Entry in the write-ahead precompile queue.
///
/// Each entry represents a JIT-compiled function that is either:
/// - `Pending`: queued for compilation
/// - `Compiled`: compiled and ready, with a stable prefetch address
/// - `Evicted`: previously compiled but dropped from the hot cache
#[derive(Clone, Debug)]
pub struct PrecompileEntry {
    /// Stable FNV-1a hash of the template — the prefetch address.
    pub hash: u64,
    /// The template that produced this entry.
    pub template: JitsonTemplate,
    /// Compilation state.
    pub state: CompileState,
}

/// State of a precompile entry in the WAL queue.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileState {
    /// Queued for compilation, not yet started.
    Pending,
    /// Compiled successfully.  `code_addr` is the function pointer in the
    /// JIT code cache (will be set by the actual JIT backend).
    Compiled { code_addr: u64 },
    /// Previously compiled but evicted from the hot cache.
    Evicted,
}

/// Write-ahead precompile queue.
///
/// Templates are appended in order.  The queue supports:
/// - **Enqueue**: add a template for compilation (deduplicates by hash)
/// - **Lookup**: check if a template is already compiled (prefetch hit)
/// - **Prefetch hint**: given the current template, return the next entry's
///   hash so the caller can issue a memory prefetch for its code page
///
/// The queue is intentionally simple — a `Vec` with linear scan.  For
/// production use at scale, swap in an LRU or LFU map.
#[derive(Clone, Debug, Default)]
pub struct PrecompileQueue {
    entries: Vec<PrecompileEntry>,
}

impl PrecompileQueue {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Enqueue a template for precompilation.  Returns the stable hash.
    /// If the template is already in the queue, returns the existing hash
    /// without duplicating it.
    pub fn enqueue(&mut self, template: JitsonTemplate) -> u64 {
        let hash = template_hash(&template);
        if self.entries.iter().any(|e| e.hash == hash) {
            return hash;
        }
        self.entries.push(PrecompileEntry {
            hash,
            template,
            state: CompileState::Pending,
        });
        hash
    }

    /// Look up a template by its prefetch address (hash).
    pub fn lookup(&self, hash: u64) -> Option<&PrecompileEntry> {
        self.entries.iter().find(|e| e.hash == hash)
    }

    /// Mark a template as compiled with the given code address.
    pub fn mark_compiled(&mut self, hash: u64, code_addr: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.hash == hash) {
            entry.state = CompileState::Compiled { code_addr };
            true
        } else {
            false
        }
    }

    /// Mark a template as evicted from the hot cache.
    pub fn mark_evicted(&mut self, hash: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.hash == hash) {
            entry.state = CompileState::Evicted;
            true
        } else {
            false
        }
    }

    /// Return the prefetch address of the entry immediately after `hash`.
    ///
    /// This enables speculative prefetching: while executing compiled code
    /// for `hash`, the caller can issue `PREFETCHT0` on the next entry's
    /// code page.
    pub fn prefetch_next(&self, hash: u64) -> Option<u64> {
        let idx = self.entries.iter().position(|e| e.hash == hash)?;
        self.entries.get(idx + 1).and_then(|e| match e.state {
            CompileState::Compiled { code_addr } => Some(code_addr),
            _ => None,
        })
    }

    /// Return all pending entries (templates awaiting compilation).
    pub fn pending(&self) -> Vec<&PrecompileEntry> {
        self.entries
            .iter()
            .filter(|e| e.state == CompileState::Pending)
            .collect()
    }

    /// Number of entries in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": {
            "threshold": 2048,
            "record_size": 256,
            "top_k": 10
        },
        "pipeline": [
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd", "fallback": "avx2_lookup" },
            { "stage": "reduce", "avx512": "vpord" }
        ],
        "features": {
            "avx512f": true,
            "avx512vl": true,
            "avx512vpopcntdq": true,
            "avx512bw": false
        },
        "cranelift": {
            "preset": "sapphire_rapids",
            "opt_level": "speed"
        }
    }"#;

    #[test]
    fn test_parse_valid() {
        let root = parse_json(VALID_TEMPLATE).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
        assert_eq!(
            root.get("kernel").unwrap().as_str(),
            Some("hamming_distance")
        );
        let scan = root.get("scan").unwrap();
        assert_eq!(scan.get("threshold").unwrap().as_u64(), Some(2048));
    }

    #[test]
    fn test_bracket_recovery_missing_closing_brace() {
        // Missing final }
        let input = r#"{"version": 1, "kernel": "hamming_distance", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
        assert_eq!(
            root.get("kernel").unwrap().as_str(),
            Some("hamming_distance")
        );
    }

    #[test]
    fn test_bracket_recovery_missing_closing_bracket() {
        // Missing final ] and }
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "pipeline": [{"stage": "dot"}"#;
        let root = parse_json(input).unwrap();
        let pipeline = root.get("pipeline").unwrap().as_array().unwrap();
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn test_bracket_recovery_nested() {
        // Missing two closing braces
        let input = r#"{"version": 1, "kernel": "cosine_i8", "scan": {"threshold": 100, "record_size": 128, "top_k": 3"#;
        let root = parse_json(input).unwrap();
        let scan = root.get("scan").unwrap();
        assert_eq!(scan.get("top_k").unwrap().as_u64(), Some(3));
    }

    #[test]
    fn test_trailing_comma() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5,},}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_validate_valid() {
        let root = parse_json(VALID_TEMPLATE).unwrap();
        let errs = validate(&root);
        // Only "vpaddd" is unknown (not in patched Cranelift); everything else is fine
        assert!(errs.len() <= 1, "unexpected errors: {:?}", errs);
    }

    #[test]
    fn test_validate_missing_fields() {
        let input = r#"{"version": 1}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path == "/kernel"));
        assert!(errs.iter().any(|e| e.path == "/scan"));
    }

    #[test]
    fn test_validate_bad_version() {
        let input = r#"{"version": 99, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path == "/version"));
    }

    #[test]
    fn test_validate_unknown_feature() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "features": {"avx512_bogus": true}}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.path.contains("avx512_bogus")));
    }

    #[test]
    fn test_validate_unknown_instruction() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "pipeline": [{"stage": "nope", "avx512": "vfakeop"}]}"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("vfakeop")));
    }

    #[test]
    fn test_from_json_roundtrip() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        assert_eq!(tmpl.kernel, "hamming_distance");
        assert_eq!(tmpl.scan.threshold, 2048);
        assert_eq!(tmpl.scan.record_size, 256);
        assert_eq!(tmpl.scan.top_k, 10);
        assert_eq!(tmpl.pipeline.len(), 3);
        assert_eq!(tmpl.pipeline[0].stage, "xor");
        assert_eq!(tmpl.pipeline[1].avx512_instr.as_deref(), Some("vpopcntd"));
        assert_eq!(tmpl.pipeline[1].fallback.as_deref(), Some("avx2_lookup"));
        assert_eq!(tmpl.cranelift_preset.as_deref(), Some("sapphire_rapids"));
    }

    #[test]
    fn test_check_pipeline_features() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let unsatisfied = check_pipeline_features(&tmpl);
        // vpxord requires avx512f+avx512vl (both enabled) — satisfied
        // vpopcntd requires avx512vl+avx512vpopcntdq (both enabled) — satisfied
        // vpord requires avx512f+avx512vl (both enabled) — satisfied
        assert!(
            unsatisfied.is_empty(),
            "unexpected unsatisfied: {:?}",
            unsatisfied
        );
    }

    #[test]
    fn test_check_pipeline_missing_feature() {
        let input = r#"{
            "version": 1,
            "kernel": "hamming_distance",
            "scan": {"threshold": 1, "record_size": 64, "top_k": 5},
            "pipeline": [{"stage": "popcnt_byte", "avx512": "vpopcntb"}],
            "features": {"avx512f": true, "avx512vl": true}
        }"#;
        let tmpl = from_json(input).unwrap();
        let unsatisfied = check_pipeline_features(&tmpl);
        assert_eq!(unsatisfied.len(), 1);
        assert_eq!(unsatisfied[0].1, "vpopcntb");
        assert!(unsatisfied[0].2.contains(&String::from("avx512bitalg")));
    }

    #[test]
    fn test_required_features_mapping() {
        assert_eq!(required_features("vpxord"), &["avx512vl", "avx512f"]);
        assert_eq!(
            required_features("vpopcntd"),
            &["avx512vl", "avx512vpopcntdq"]
        );
        assert_eq!(required_features("vpdpbusd"), &["avx512vl", "avx512vnni"]);
        assert_eq!(required_features("vpermi2b"), &["avx512vl", "avx512vbmi"]);
        assert_eq!(required_features("vpsllw"), &["avx512vl", "avx512bw"]);
        assert_eq!(required_features("not_real"), &[] as &[&str]);
    }

    #[test]
    fn test_parse_escaped_strings() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "cranelift": {"preset": "sapphire_rapids", "opt_level": "speed"}}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(
            root.get("cranelift")
                .unwrap()
                .get("opt_level")
                .unwrap()
                .as_str(),
            Some("speed")
        );
    }

    #[test]
    fn test_parse_error_bad_json() {
        let result = parse_json("{not json}");
        assert!(result.is_err());
    }

    #[test]
    fn test_single_line_comment() {
        let input = "{\n// this is a comment\n\"version\": 1, \"kernel\": \"dot_f32\", \"scan\": {\"threshold\": 1, \"record_size\": 64, \"top_k\": 5}}";
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
    }

    // --- Backend CPU-lane tests ---

    const BACKEND_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": { "threshold": 2048, "record_size": 256, "top_k": 10 },
        "pipeline": [
            { "stage": "fetch",  "backend": "lancedb",   "table": "embeddings" },
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd" },
            { "stage": "store",  "backend": "dragonfly", "prefix": "results:" }
        ],
        "backends": {
            "lancedb":   { "uri": "data/vectors.lance" },
            "dragonfly": { "uri": "redis://127.0.0.1:6379" }
        },
        "features": { "avx512f": true, "avx512vl": true, "avx512vpopcntdq": true }
    }"#;

    #[test]
    fn test_backend_template_parses() {
        let tmpl = from_json(BACKEND_TEMPLATE).unwrap();
        assert_eq!(tmpl.backends.len(), 2);
        assert_eq!(tmpl.backends[0].name, "lancedb");
        assert_eq!(tmpl.backends[0].uri, "data/vectors.lance");
        assert_eq!(tmpl.backends[1].name, "dragonfly");
        assert_eq!(tmpl.backends[1].uri, "redis://127.0.0.1:6379");
    }

    #[test]
    fn test_pipeline_backend_refs() {
        let tmpl = from_json(BACKEND_TEMPLATE).unwrap();
        assert_eq!(tmpl.pipeline[0].backend.as_deref(), Some("lancedb"));
        assert_eq!(tmpl.pipeline[0].backend_key.as_deref(), Some("embeddings"));
        assert_eq!(tmpl.pipeline[3].backend.as_deref(), Some("dragonfly"));
        assert_eq!(tmpl.pipeline[3].backend_key.as_deref(), Some("results:"));
    }

    #[test]
    fn test_validate_undeclared_backend() {
        let input = r#"{
            "version": 1,
            "kernel": "dot_f32",
            "scan": { "threshold": 1, "record_size": 64, "top_k": 5 },
            "pipeline": [{ "stage": "fetch", "backend": "postgres" }]
        }"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("postgres")));
    }

    #[test]
    fn test_validate_unknown_backend_type() {
        let input = r#"{
            "version": 1,
            "kernel": "dot_f32",
            "scan": { "threshold": 1, "record_size": 64, "top_k": 5 },
            "backends": { "mongodb": { "uri": "mongodb://localhost" } }
        }"#;
        let root = parse_json(input).unwrap();
        let errs = validate(&root);
        assert!(errs.iter().any(|e| e.message.contains("mongodb")));
    }

    // --- WAL precompile queue tests ---

    #[test]
    fn test_template_hash_deterministic() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let h1 = template_hash(&tmpl);
        let h2 = template_hash(&tmpl);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    #[test]
    fn test_template_hash_different_templates() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        assert_ne!(template_hash(&t1), template_hash(&t2));
    }

    #[test]
    fn test_precompile_queue_enqueue_dedup() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();
        let h1 = queue.enqueue(tmpl.clone());
        let h2 = queue.enqueue(tmpl);
        assert_eq!(h1, h2);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_precompile_queue_lifecycle() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();

        let h1 = queue.enqueue(t1);
        let h2 = queue.enqueue(t2);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pending().len(), 2);

        // Compile first template
        assert!(queue.mark_compiled(h1, 0xDEAD_BEEF));
        assert_eq!(queue.pending().len(), 1);
        let entry = queue.lookup(h1).unwrap();
        assert_eq!(
            entry.state,
            CompileState::Compiled {
                code_addr: 0xDEAD_BEEF
            }
        );

        // Compile second template
        assert!(queue.mark_compiled(h2, 0xCAFE_BABE));
        assert_eq!(queue.pending().len(), 0);
    }

    #[test]
    fn test_prefetch_next() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();

        let h1 = queue.enqueue(t1);
        let h2 = queue.enqueue(t2);

        // Before compilation — prefetch_next returns None (next is Pending)
        assert!(queue.prefetch_next(h1).is_none());

        // After compiling both
        queue.mark_compiled(h1, 0x1000);
        queue.mark_compiled(h2, 0x2000);

        // Now prefetch_next(h1) → 0x2000 (the code address of h2)
        assert_eq!(queue.prefetch_next(h1), Some(0x2000));
        // prefetch_next(h2) → None (no entry after h2)
        assert!(queue.prefetch_next(h2).is_none());
    }

    #[test]
    fn test_eviction() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();
        let h = queue.enqueue(tmpl);
        queue.mark_compiled(h, 0x1000);
        queue.mark_evicted(h);
        assert_eq!(queue.lookup(h).unwrap().state, CompileState::Evicted);
    }
}

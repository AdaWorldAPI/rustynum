# =============================================================================
# rustynum-oracle — Docker build (stable Rust 1.93 + AVX-512)
# =============================================================================
# All SIMD uses stable std::arch via simd_compat — no nightly required.
#
# BUILD:   docker build -t rustynum-oracle .
# RUN:     docker run rustynum-oracle ghost_oracle [path/to/graph.json]
# RAILWAY: Auto-detects Dockerfile → builds with stable + AVX-512
# =============================================================================

# =============================================================================
# STAGE 1: Builder
# =============================================================================
FROM rust:1.93-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# --- Dependency caching: copy manifests first ---
COPY Cargo.toml Cargo.lock* ./
COPY rustynum-core/Cargo.toml rustynum-core/Cargo.toml
COPY rustynum-oracle/Cargo.toml rustynum-oracle/Cargo.toml
COPY rustynum-rs/Cargo.toml rustynum-rs/Cargo.toml
COPY rustyblas/Cargo.toml rustyblas/Cargo.toml
COPY rustymkl/Cargo.toml rustymkl/Cargo.toml
COPY rustynum-holo/Cargo.toml rustynum-holo/Cargo.toml
COPY rustynum-focus/Cargo.toml rustynum-focus/Cargo.toml
COPY rustynum-carrier/Cargo.toml rustynum-carrier/Cargo.toml
COPY rustynum-arrow/Cargo.toml rustynum-arrow/Cargo.toml
COPY rustynum-clam/Cargo.toml rustynum-clam/Cargo.toml
COPY rustynum-archive/Cargo.toml rustynum-archive/Cargo.toml
COPY rustynum-archive-v3/Cargo.toml rustynum-archive-v3/Cargo.toml
COPY bindings/python/Cargo.toml bindings/python/Cargo.toml

# Create stub lib.rs so cargo fetch resolves deps
RUN for d in rustynum-core rustynum-oracle rustynum-rs rustyblas rustymkl \
             rustynum-holo rustynum-focus rustynum-carrier rustynum-arrow \
             rustynum-clam rustynum-archive rustynum-archive-v3; do \
        mkdir -p "$d/src" && echo "pub fn _stub() {}" > "$d/src/lib.rs"; \
    done && \
    mkdir -p bindings/python/src && echo "pub fn _stub() {}" > bindings/python/src/lib.rs && \
    cargo fetch 2>/dev/null || true

# --- Full source ---
COPY . .

# --- Build with AVX-512 ---
RUN RUSTFLAGS="-C target-cpu=x86-64-v4 -C link-arg=-s" \
    cargo build --release --package rustynum-oracle

# Collect binaries
RUN mkdir -p /build/out && \
    for bin in ghost_oracle ghost_discovery sweep_runner sweetspot \
               organic_sweep analysis recognize stress_test; do \
        cp "target/release/$bin" /build/out/ 2>/dev/null || true; \
    done

# =============================================================================
# STAGE 2: Runtime (minimal)
# =============================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash oracle

COPY --from=builder /build/out/ /usr/local/bin/
COPY --from=builder /build/rustynum-oracle/data/ /home/oracle/data/

USER oracle
WORKDIR /home/oracle

ENTRYPOINT ["ghost_oracle"]

//! AI War Ghost Oracle runner.
//!
//! Usage: cargo run --bin ghost_oracle [path/to/aiwar_graph.json]
//! Default path: data/aiwar_graph.json

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "data/aiwar_graph.json".to_string());

    rustynum_oracle::aiwar_ghost::run_aiwar_ghost_oracle(&path);
}

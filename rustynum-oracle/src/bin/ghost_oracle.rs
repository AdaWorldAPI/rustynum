//! AI War Ghost Oracle runner.
//!
//! Usage: cargo run --bin ghost_oracle [path/to/aiwar_graph.json]
//! Default: searches multiple paths for aiwar_graph.json

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        // Search multiple candidate paths (Railway, local dev, workspace root)
        let candidates = [
            "data/aiwar_graph.json",
            "rustynum-oracle/data/aiwar_graph.json",
            "../rustynum-oracle/data/aiwar_graph.json",
            "/home/oracle/data/aiwar_graph.json",
        ];
        for candidate in &candidates {
            if std::path::Path::new(candidate).exists() {
                return candidate.to_string();
            }
        }
        eprintln!("ERROR: Could not find aiwar_graph.json");
        eprintln!("Searched: {:?}", candidates);
        eprintln!("Working directory: {:?}", std::env::current_dir().ok());
        eprintln!("Provide path as argument: ghost_oracle /path/to/aiwar_graph.json");
        std::process::exit(1);
    });

    rustynum_oracle::aiwar_ghost::run_aiwar_ghost_oracle(&path);
}

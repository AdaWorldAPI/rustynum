// Stroke 1 — Prefix XOR + popcount (Belichtungsmesser GPU kernel)
//
// Each invocation: ONE candidate's prefix XOR+popcount.
// Zero branching. Pure throughput.
// Early-exit decisions happen on CPU after readback.

struct Params {
    prefix_words: u32,
    num_candidates: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> query_prefix: array<u32>;
@group(0) @binding(2) var<storage, read> database: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

@compute @workgroup_size(64)
fn stroke1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let candidate_id = gid.x;
    if (candidate_id >= params.num_candidates) {
        return;
    }

    let base = candidate_id * params.prefix_words;
    var dist: u32 = 0u;

    let full_iters = params.prefix_words / 4u;
    let remainder = params.prefix_words % 4u;

    for (var i: u32 = 0u; i < full_iters; i = i + 1u) {
        let off = i * 4u;
        let x0 = query_prefix[off]      ^ database[base + off];
        let x1 = query_prefix[off + 1u] ^ database[base + off + 1u];
        let x2 = query_prefix[off + 2u] ^ database[base + off + 2u];
        let x3 = query_prefix[off + 3u] ^ database[base + off + 3u];

        dist = dist
            + countOneBits(x0)
            + countOneBits(x1)
            + countOneBits(x2)
            + countOneBits(x3);
    }

    let rem_base = full_iters * 4u;
    for (var i: u32 = 0u; i < remainder; i = i + 1u) {
        dist = dist + countOneBits(
            query_prefix[rem_base + i] ^ database[base + rem_base + i]
        );
    }

    results[candidate_id] = dist;
}

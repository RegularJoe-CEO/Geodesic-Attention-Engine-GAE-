// Waller Operator v2 - Tiled for Apple Silicon M1 Pro
// Each workgroup: one (position, head) pair
// 64 threads cooperatively compute attention for one output row

struct Params {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    num_heads: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;  
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const TILE_K: u32 = 32u;

var<workgroup> tile_k: array<f32, 2048>;
var<workgroup> tile_v: array<f32, 2048>;
var<workgroup> shared_scores: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let pos = wid.x;
    let tid = lid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let scale = params.scale;
    
    if (pos >= seq_len) { return; }
    
    var q_reg: array<f32, 64>;
    for (var d: u32 = 0u; d < head_dim; d++) {
        q_reg[d] = q[pos * head_dim + d];
    }
    
    var m: f32 = -3.402823e+38;
    var l: f32 = 0.0;
    var acc: array<f32, 64>;
    for (var d: u32 = 0u; d < head_dim; d++) {
        acc[d] = 0.0;
    }
    
    let num_keys = pos + 1u;
    let num_tiles = (num_keys + TILE_K - 1u) / TILE_K;
    
    for (var tile: u32 = 0u; tile < num_tiles; tile++) {
        let tile_start = tile * TILE_K;
        let tile_end = min(tile_start + TILE_K, num_keys);
        let tile_size = tile_end - tile_start;
        
        for (var i: u32 = tid; i < tile_size * head_dim; i += 64u) {
            let local_j = i / head_dim;
            let d = i % head_dim;
            let global_j = tile_start + local_j;
            tile_k[local_j * head_dim + d] = k[global_j * head_dim + d];
            tile_v[local_j * head_dim + d] = v[global_j * head_dim + d];
        }
        workgroupBarrier();
        
        for (var local_j: u32 = 0u; local_j < tile_size; local_j++) {
            var score: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d++) {
                score += q_reg[d] * tile_k[local_j * head_dim + d];
            }
            score *= scale;
            
            let m_new = max(m, score);
            let correction = exp(m - m_new);
            let weight = exp(score - m_new);
            
            l = l * correction + weight;
            for (var d: u32 = 0u; d < head_dim; d++) {
                acc[d] = acc[d] * correction + weight * tile_v[local_j * head_dim + d];
            }
            m = m_new;
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        let inv_l = 1.0 / l;
        for (var d: u32 = 0u; d < head_dim; d++) {
            output[pos * head_dim + d] = acc[d] * inv_l;
        }
    }
}

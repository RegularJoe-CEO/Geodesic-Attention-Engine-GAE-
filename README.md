# Geodesic Attention Engine (GAE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18512336.svg)](https://doi.org/10.5281/zenodo.18512336)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)


](https://www.gnu.org/licenses/agpl-3.0)

My name is Eric Waller, and it's my hope that this project helps.

GAE computes exact transformer attention with fewer memory operations. The Fused Waller Kernel reduces HBM round-trips from 12 to 2, achieving O(N) memory complexity instead of O(N²).

📄 Full Technical Specification — Deep dive into the math and implementation

Quick Results
On NVIDIA H100 80GB:

Sequence Length	Standard Attention Memory	GAE Memory	Reduction
65,536	17.25 GB	0.62 GB	99.6%
262,144	275 GB (impossible)	0.82 GB	✓ Works
1,048,576	4.4 TB (impossible)	1.09 GB	✓ Works
GAE enables 1M+ token sequences on hardware that can't fit 64K with standard attention.

How It Works
Standard attention:



Load Q → Compute → Store → Load K → Compute → Store → Load V → ... (12 HBM trips)
GAE:



Load Q,K,V → Compute Everything in Registers → Store Output (2 HBM trips)
Key techniques:

Online Softmax — Single streaming pass, no O(N²) intermediate matrix
Register-Level Fusion — Q·Kᵀ, softmax, ×V all in registers
Welford Statistics — Numerically stable, bit-exact determinism
Installation
bash


git clone https://github.com/RegularJoe-CEO/Geodesic-Attention-Engine-GAE-.git
cd Geodesic-Attention-Engine-GAE-
cargo build --release
Requirements: CUDA 11.8+, Rust 1.70+, NVIDIA GPU (Ampere+)

Benchmarks
Full benchmark results with exact commands: benches/BENCHMARK_RESULTS.md

Run Benchmarks Yourself
Rust reference implementation:

bash


cargo bench
CUDA O(1) Memory Kernel (production):

bash


cd cuda_src
nvcc -O3 -arch=sm_90 waller_operator.cu -o waller_bench
./waller_bench
CUDA Tiled cuBLAS Kernel (tensor cores):

bash


cd cuda_src
nvcc -O3 -arch=sm_90 waller_v7.cu -lcublas -o waller_v7_bench
./waller_v7_bench
INT8 Quantized Kernel:

bash


cd cuda_src
nvcc -O3 -arch=sm_90 waller_operator_int8_tiled.cu -o int8_bench
./int8_bench
Minimal Usage Example
rust


use gae::{WallerKernel, AttentionConfig};

let config = AttentionConfig {
    seq_len: 4096,
    head_dim: 64,
    num_heads: 12,
};

let kernel = WallerKernel::new(&config)?;
let output = kernel.forward(&q, &k, &v);  // 2 HBM trips, bit-exact
See benches/usage_example.rs for complete example.

Backends
Backend	Status	Notes
CUDA	Production	Tested on A100/H100
Rust	Reference	Pure implementation
WebGPU	Experimental	Browser support
What GAE Is Not
Not approximate — Computes exact attention, every query attends to every key
Not sparse — Full attention matrix semantics
Not a FlashAttention replacement — FlashAttention has broader ecosystem support; GAE demonstrates further fusion is possible
Citation


@software{waller2026gae,
  author = {Waller, Eric},
  title = {Geodesic Attention Engine: Fused O(N) Exact Attention},
  year = {2026},
  doi = {10.5281/zenodo.18512336},
  url = {https://github.com/RegularJoe-CEO/Geodesic-Attention-Engine-GAE-}
}
License
AGPL-3.0 — See LICENSE

Contact
Eric Waller
e@ewaller.com
https://luxiedge.com

© 2026 Eric Waller — Patent Pending

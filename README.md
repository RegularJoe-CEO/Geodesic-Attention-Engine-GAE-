# Geodesic Attention Engine (GAE)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18512336.svg)](https://doi.org/10.5281/zenodo.18512336)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

GAE computes **exact transformer attention** with fewer memory operations. The Fused Waller Kernel reduces HBM round-trips from 12 to 2, achieving **O(N) memory complexity** instead of O(N²).

> **[Full Technical Specification](TECHNICAL_SPEC.md)** — Deep dive into the math and implementation

---

## Quick Results on NVIDIA H100 80GB

| Sequence Length | Standard Attention | GAE Memory | Reduction |
|-----------------|-------------------|------------|-----------|
| 65,536 | 17.25 GB | 0.62 GB | 99.6% |
| 262,144 | 275 GB (impossible) | 0.82 GB | ✓ Works |
| 1,048,576 | 4.4 TB (impossible) | 1.09 GB | ✓ Works |

**GAE enables 1M+ token sequences on hardware that cannot fit 64K with standard attention.**

---

## How It Works

**Standard attention:** 12 HBM round-trips  
**GAE:** 2 HBM round-trips (Load Q,K,V → Compute in Registers → Store Output)

**Key techniques:**
- **Online Softmax** — Single streaming pass, no O(N²) intermediate matrix
- **Register-Level Fusion** — Q·Kᵀ, softmax, ×V all in registers
- **Welford Statistics** — Numerically stable, bit-exact determinism

---

## Energy Efficiency Benchmarks

**Energy ∝ Data Movement** — The Waller Kernel eliminates N² memory traffic.

| Benchmark | What It Measures | Command |
|-----------|------------------|---------|
| `energy_thesis.rs` | Memory traffic analysis, 75% energy reduction proof | `cargo run --bin energy_thesis` |
| `energy_benchmark.rs` | GFLOPS, efficiency rating across sequence lengths | `cargo run --bin energy_benchmark` |
| `gpu_energy_benchmark.rs` | CPU vs GPU speedup, GPU GFLOPS | `cargo run --bin gpu_energy_benchmark` |

At 8K+ context: **>99% memory reduction**, **>75% energy savings**.

Full benchmark results: **[benches/BENCHMARK_RESULTS.md](benches/BENCHMARK_RESULTS.md)**

---

## Installation

```bash
git clone https://github.com/RegularJoe-CEO/Geodesic-Attention-Engine-GAE-.git
cd Geodesic-Attention-Engine-GAE-
cargo build --release
Requirements: CUDA 11.8+, Rust 1.70+, NVIDIA GPU (Ampere+)

Run Benchmarks
Rust:

bash


cargo bench
CUDA O(1) Memory Kernel:

bash


cd cuda_src && nvcc -O3 -arch=sm_90 waller_operator.cu -o bench && ./bench
CUDA cuBLAS Kernel:

bash


cd cuda_src && nvcc -O3 -arch=sm_90 waller_v7.cu -lcublas -o bench && ./bench
Backends
Backend	Status	Notes
CUDA	Production	A100/H100
Rust	Reference	Pure impl
WebGPU	Experimental	Browser
What GAE Is Not
Not approximate — Exact attention, bit-for-bit identical results
Not sparse — Full attention matrix semantics
Not a FlashAttention replacement — Different approach, demonstrates further fusion possibilities
License
AGPL-3.0

Contact
Eric Waller — e@ewaller.com — https://luxiedge.com

© 2026 Eric Waller

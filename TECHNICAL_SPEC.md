# Geodesic Attention Engine (GAE): Technical Specification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18512336.svg)](https://doi.org/10.5281/zenodo.18512336)

## What GAE Does

GAE computes exact transformer attention with fewer memory operations. It fuses the Q·Kᵀ multiplication, softmax normalization, and ×V multiplication into a single GPU kernel—the Fused Waller Kernel—that keeps all intermediates in registers instead of writing them to HBM.

The result: 83% reduction in HBM round-trips (12 → 2), O(N) memory complexity instead of O(N²), and 23-37% improvement in Tok/J on H100 GPUs.

This is not approximate attention. Every query attends to every key. The math is identical to standard attention—just computed with less memory movement.

## The Problem

Attention is memory-bound, not compute-bound.

Standard attention does this:

1. Load Q from HBM → Compute Q·Kᵀ → Write attention scores to HBM
2. Load scores from HBM → Compute softmax → Write normalized scores to HBM
3. Load normalized scores from HBM → Compute ×V → Write output to HBM

Each arrow is a memory transfer. HBM bandwidth on an H100 is ~3.35 TB/s. Register bandwidth is effectively infinite (same-cycle access). Every HBM round-trip costs time and energy.

For a single attention pass, standard implementations touch HBM 12 times. Most of those trips are unnecessary—they exist because the operations were written as separate kernels.

## The Solution

### The Fused Waller Kernel

The Fused Waller Kernel combines Q·Kᵀ, softmax, and ×V into one kernel launch:

1. Load Q, K, V from HBM once
2. Compute Q·Kᵀ in registers
3. Compute softmax in registers (online algorithm)
4. Compute ×V in registers
5. Write output to HBM once

Two HBM accesses. Everything else stays in registers.

### Register-Level Fusion

The key constraint: GPU registers are scarce (~256KB per SM on H100). The kernel must tile the computation so that each tile fits entirely in registers.

For each tile:
- Q block stays resident
- K, V blocks stream through
- Partial softmax statistics accumulate via Welford's online algorithm
- Output accumulates without materializing the full attention matrix

The attention matrix is never fully materialized. O(N²) storage becomes O(N).

### Online Softmax

Standard softmax requires two passes: one to find the max (for numerical stability), one to compute exp and normalize. This requires storing intermediate values.

Online softmax computes the max, exponentials, and normalization in a single streaming pass using running statistics. No intermediate storage. Numerically stable. Bit-exact with standard softmax.

## The Math

For verification, GAE computes standard attention:
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V




Where softmax is:
softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))




The Fused Waller Kernel produces bit-exact results. Any difference indicates a bug, not an approximation tradeoff.

## Results

Measured on H100 SXM, sequence length 4096, head dimension 64:

| Metric | Standard | Fused Waller Kernel | Change |
|--------|----------|---------------------|--------|
| HBM Round-trips | 12 | 2 | -83% |
| Memory Complexity | O(N²) | O(N) | Linear |
| Tok/J | baseline | +23-37% | Measured |
| Determinism | Variable | Bit-exact | Reproducible |

## How to Reproduce

### Requirements
- NVIDIA GPU (Ampere or newer recommended)
- CUDA 11.8+
- Rust 1.70+

### Build
```bash
git clone https://github.com/RegularJoe-CEO/Geodesic-Attention-Engine-GAE-.git
cd Geodesic-Attention-Engine-GAE-
cargo build --release
Run Benchmarks
bash


cargo bench
Verify Correctness
bash


cargo test
This runs bit-exact comparison against reference attention. All tests must pass.

Backends
CUDA — Production, tested on A100/H100
Rust — Reference implementation
WebGPU — Experimental
What GAE Is Not
GAE is not an approximation—it computes exact attention. Not sparse—every query attends to every key. Not a FlashAttention replacement; FlashAttention has broader features and support. GAE demonstrates that further fusion is possible.

Citation


Waller, E. (2026). Geodesic Attention Engine (GAE): Minimum-Energy Path Through Transformer Attention. Zenodo. https://doi.org/10.5281/zenodo.18512336
License
GNU GPL v3.0

Contact
Eric Waller
e@ewaller.com
https://luxiedge.com




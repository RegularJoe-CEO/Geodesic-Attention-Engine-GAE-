# Geodesic Attention Engine (GAE)

My name is Eric Waller, and it's my hope that this project helps.

GAE is a fused attention implementation that finds the shortest path through the attention computation—reducing memory traffic by eliminating unnecessary round-trips to high-bandwidth memory (HBM).

## What GAE Does

Standard transformer attention implementations read and write to HBM multiple times during a single attention computation. Each trip costs energy and time. GAE fuses the entire attention operation—Q, K, V projection, softmax, and output—into a single kernel that touches HBM only twice: once to load, once to store.

**The core insight:** Most attention implementations optimize individual operations. GAE optimizes the path between them.

## How It Works

### The Fused Waller Operator

Traditional attention requires 12 HBM round-trips:
Load Q, Compute, Store, Load K, Compute, Store, Load V, Compute, Store, Load QK^T, Softmax, Store, Load, Apply V, Store, Output

GAE requires 2 HBM round-trips:
Load Q, K, V → Compute Everything in Registers → Store Output

### Key Techniques

1. **Online Softmax**: Computes softmax in a single streaming pass without materializing the full attention matrix. Reduces memory complexity from O(N²) to O(N).

2. **Welford Streaming Statistics**: Maintains running mean and variance with numerical stability, enabling bit-exact deterministic results across runs.

3. **Register-Level Fusion**: Q, K, V, intermediate products, and softmax all stay in registers. Nothing hits HBM until the final output.

## Measured Results

Benchmarks on NVIDIA H100, sequence length 4096, head dimension 64:

| Metric | Standard | GAE | Change |
|--------|----------|-----|--------|
| HBM Round-trips | 12 | 2 | -83% |
| Memory Complexity | O(N²) | O(N) | Linear |
| Energy per Token | Baseline | -23% to -37% | Tok/J |
| Determinism | Variable | Bit-exact | Reproducible |

## Installation

Clone and build with Cargo:

    git clone https://github.com/RegularJoe-CEO/Geodesic-Attention-Engine-GAE-.git
    cd Geodesic-Attention-Engine-GAE-
    cargo build --release

Requirements: CUDA 11.8+, Rust 1.70+, NVIDIA GPU (Ampere+)

## Backends

- **CUDA** — Production, tested on A100/H100
- **Rust** — Reference implementation
- **WebGPU** — Experimental

## What GAE Is Not

GAE is not an approximation—it computes exact attention. Not sparse—every query attends to every key. Not a FlashAttention replacement in all cases; FlashAttention has more features and broader support. GAE demonstrates further fusion is possible.

## Citation

See repository for citation format.

## License

AGPL-3.0

## Contact

Eric Waller
e@ewaller.com
https://luxiedge.com

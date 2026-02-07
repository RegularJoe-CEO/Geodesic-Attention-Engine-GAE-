//! GAE Attention Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ate::{waller_operator, GAEConfig};

fn bench_waller_operator(c: &mut Criterion) {
    let config = GAEConfig::gpt2_small();
    let seq_len = 512;
    let head_dim = config.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Random test data
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    c.bench_function("waller_512_seq", |b| {
        b.iter(|| {
            waller_operator(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(seq_len),
                black_box(head_dim),
                black_box(scale),
            )
        })
    });
}

criterion_group!(benches, bench_waller_operator);
criterion_main!(benches, parallel_benches, gpu_benches, large_benches, huge_benches);

fn bench_waller_parallel(c: &mut Criterion) {
    let config = ate::GAEConfig::gpt2_small();
    let seq_len = 512;
    let head_dim = config.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    c.bench_function("waller_512_parallel", |b| {
        b.iter(|| {
            ate::waller_operator_parallel(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(seq_len),
                black_box(head_dim),
                black_box(scale),
            )
        })
    });
}

fn bench_full_transformer(c: &mut Criterion) {
    let config = ate::TransformerConfig {
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        mlp_dim: 256,
        eps: 1e-5,
    };
    let block = ate::TransformerBlock::new_random(config);
    let seq_len = 128;
    let input: Vec<f32> = (0..seq_len * 64).map(|i| (i as f32 * 0.01).sin()).collect();
    
    c.bench_function("transformer_128_seq", |b| {
        b.iter(|| {
            block.forward(black_box(&input), black_box(seq_len))
        })
    });
}

criterion_group!(parallel_benches, bench_waller_parallel, bench_full_transformer);

#[cfg(feature = "wgpu")]
fn bench_waller_gpu(c: &mut Criterion) {
    let config = ate::GAEConfig::gpt2_small();
    let seq_len = 512;
    let head_dim = config.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    let gpu = ate::gpu::AteGpu::new();
    
    // Warmup
    let _ = gpu.waller_operator(&q, &k, &v, seq_len, head_dim, scale);
    
    c.bench_function("waller_512_gpu", |b| {
        b.iter(|| {
            gpu.waller_operator(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(seq_len),
                black_box(head_dim),
                black_box(scale),
            )
        })
    });
}

#[cfg(feature = "wgpu")]
criterion_group!(gpu_benches, bench_waller_gpu);

#[cfg(feature = "wgpu")]
fn bench_waller_gpu_large(c: &mut Criterion) {
    let seq_len = 2048;
    let head_dim = 128; // Larger head dim
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    let gpu = ate::gpu::AteGpu::new();
    let _ = gpu.waller_operator(&q, &k, &v, seq_len, head_dim, scale); // warmup
    
    c.bench_function("waller_2048_gpu", |b| {
        b.iter(|| {
            gpu.waller_operator(
                black_box(&q), black_box(&k), black_box(&v),
                black_box(seq_len), black_box(head_dim), black_box(scale),
            )
        })
    });
}

fn bench_waller_large_parallel(c: &mut Criterion) {
    let seq_len = 2048;
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    c.bench_function("waller_2048_parallel", |b| {
        b.iter(|| {
            ate::waller_operator_parallel(
                black_box(&q), black_box(&k), black_box(&v),
                black_box(seq_len), black_box(head_dim), black_box(scale),
            )
        })
    });
}

#[cfg(feature = "wgpu")]
criterion_group!(large_benches, bench_waller_gpu_large, bench_waller_large_parallel);

#[cfg(not(feature = "wgpu"))]
criterion_group!(large_benches, bench_waller_large_parallel);

#[cfg(feature = "wgpu")]
fn bench_waller_gpu_4k(c: &mut Criterion) {
    let seq_len = 4096;
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    let gpu = ate::gpu::AteGpu::new();
    let _ = gpu.waller_operator(&q, &k, &v, seq_len, head_dim, scale);
    
    c.bench_function("waller_4096_gpu", |b| {
        b.iter(|| {
            gpu.waller_operator(
                black_box(&q), black_box(&k), black_box(&v),
                black_box(seq_len), black_box(head_dim), black_box(scale),
            )
        })
    });
}

fn bench_waller_4k_parallel(c: &mut Criterion) {
    let seq_len = 4096;
    let head_dim = 128;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();
    
    c.bench_function("waller_4096_parallel", |b| {
        b.iter(|| {
            ate::waller_operator_parallel(
                black_box(&q), black_box(&k), black_box(&v),
                black_box(seq_len), black_box(head_dim), black_box(scale),
            )
        })
    });
}

#[cfg(feature = "wgpu")]
criterion_group!(huge_benches, bench_waller_gpu_4k, bench_waller_4k_parallel);

#[cfg(not(feature = "wgpu"))]
criterion_group!(huge_benches, bench_waller_4k_parallel);

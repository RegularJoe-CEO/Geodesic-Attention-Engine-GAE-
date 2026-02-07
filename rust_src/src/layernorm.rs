//! LayerNorm using Welford streaming statistics

use crate::welford::WelfordState;

/// Single-pass LayerNorm using Welford algorithm
pub fn layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let mut state = WelfordState::new();
    for &x in input {
        state.update(x);
    }
    let mean = state.mean;
    let std = state.std(eps);
    
    input.iter().zip(gamma.iter()).zip(beta.iter())
        .map(|((&x, &g), &b)| (x - mean) / std * g + b)
        .collect()
}

/// Batched LayerNorm
pub fn layernorm_batched(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    
    for b in 0..batch_size {
        let start = b * hidden_dim;
        let end = start + hidden_dim;
        let row = &input[start..end];
        
        let mut state = WelfordState::new();
        for &x in row {
            state.update(x);
        }
        let mean = state.mean;
        let std = state.std(eps);
        
        for i in 0..hidden_dim {
            output[start + i] = (row[i] - mean) / std * gamma[i] + beta[i];
        }
    }
    output
}

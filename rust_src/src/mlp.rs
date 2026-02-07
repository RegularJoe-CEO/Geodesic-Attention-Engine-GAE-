//! Fused MLP block with LayerNorm

use crate::activations::gelu;
use crate::welford::WelfordState;

/// Basic MLP block: Linear -> GELU -> Linear
pub fn mlp_block(
    input: &[f32],
    w1: &[f32],      // [hidden_dim, mlp_dim]
    b1: &[f32],      // [mlp_dim]
    w2: &[f32],      // [mlp_dim, hidden_dim]
    b2: &[f32],      // [hidden_dim]
    hidden_dim: usize,
    mlp_dim: usize,
) -> Vec<f32> {
    // First linear: input @ W1 + b1
    let mut intermediate = vec![0.0; mlp_dim];
    for i in 0..mlp_dim {
        let mut sum = b1[i];
        for j in 0..hidden_dim {
            sum += input[j] * w1[j * mlp_dim + i];
        }
        intermediate[i] = gelu(sum);
    }
    
    // Second linear: intermediate @ W2 + b2
    let mut output = vec![0.0; hidden_dim];
    for i in 0..hidden_dim {
        let mut sum = b2[i];
        for j in 0..mlp_dim {
            sum += intermediate[j] * w2[j * hidden_dim + i];
        }
        output[i] = sum;
    }
    output
}

/// Fused MLP + LayerNorm + Residual (single pass stats)
pub fn fused_mlp_layernorm(
    input: &[f32],
    residual: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden_dim: usize,
    mlp_dim: usize,
    eps: f32,
) -> Vec<f32> {
    // MLP forward
    let mlp_out = mlp_block(input, w1, b1, w2, b2, hidden_dim, mlp_dim);
    
    // Residual add
    let mut combined: Vec<f32> = mlp_out.iter()
        .zip(residual.iter())
        .map(|(&m, &r)| m + r)
        .collect();
    
    // Fused LayerNorm using Welford
    let mut state = WelfordState::new();
    for &x in &combined {
        state.update(x);
    }
    let mean = state.mean;
    let std = state.std(eps);
    
    for i in 0..hidden_dim {
        combined[i] = (combined[i] - mean) / std * gamma[i] + beta[i];
    }
    combined
}

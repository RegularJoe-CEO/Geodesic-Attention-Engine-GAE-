//! Validation tests for GAE correctness

use ate::*;
use approx::assert_relative_eq;

#[test]
fn test_welford_mean_variance() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let mut state = WelfordState::new();
    for &x in &data {
        state.update(x);
    }
    assert_relative_eq!(state.mean, 5.0, epsilon = 1e-5);
    assert_relative_eq!(state.variance(), 4.0, epsilon = 1e-5);
}

#[test]
fn test_welford_merge() {
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    
    let mut state1 = WelfordState::new();
    for &x in &data1 { state1.update(x); }
    
    let mut state2 = WelfordState::new();
    for &x in &data2 { state2.update(x); }
    
    let merged = WelfordState::merge(&state1, &state2);
    
    let mut full = WelfordState::new();
    for &x in &data1 { full.update(x); }
    for &x in &data2 { full.update(x); }
    
    assert_relative_eq!(merged.mean, full.mean, epsilon = 1e-5);
    assert_relative_eq!(merged.variance(), full.variance(), epsilon = 1e-5);
}

#[test]
fn test_online_softmax_equivalence() {
    let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    let mut online = OnlineSoftmax::new();
    for &s in &scores {
        online.update(s);
    }
    
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
    
    for &s in &scores {
        let online_prob = online.probability(s);
        let standard_prob = (s - max_score).exp() / exp_sum;
        assert_relative_eq!(online_prob, standard_prob, epsilon = 1e-5);
    }
}

#[test]
fn test_gelu_values() {
    assert_relative_eq!(gelu(0.0), 0.0, epsilon = 1e-5);
    assert_relative_eq!(gelu(1.0), 0.8413, epsilon = 1e-3);
    assert_relative_eq!(gelu(-1.0), -0.1587, epsilon = 1e-3);
}

#[test]
fn test_layernorm_unit() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    
    let output = layernorm(&input, &gamma, &beta, 1e-5);
    
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    let var: f32 = output.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
    
    assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
    assert_relative_eq!(var, 1.0, epsilon = 1e-2);
}

#[test]
fn test_waller_operator_causal() {
    let seq_len = 4;
    let head_dim = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let q = vec![1.0; seq_len * head_dim];
    let k = vec![1.0; seq_len * head_dim];
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
    
    let output = waller_operator(&q, &k, &v, seq_len, head_dim, scale);
    
    assert_eq!(output.len(), seq_len * head_dim);
    assert_relative_eq!(output[0], v[0], epsilon = 1e-4);
    assert_relative_eq!(output[1], v[1], epsilon = 1e-4);
}

#[test]
fn test_transformer_block_shapes() {
    let config = TransformerConfig {
        hidden_dim: 64,
        num_heads: 4,
        head_dim: 16,
        mlp_dim: 256,
        eps: 1e-5,
    };
    
    let block = TransformerBlock::new_random(config);
    let seq_len = 8;
    let input = vec![0.1; seq_len * 64];
    
    let output = block.forward(&input, seq_len);
    
    assert_eq!(output.len(), input.len());
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_ate_config_presets() {
    let gpt2 = GAEConfig::gpt2_small();
    assert_eq!(gpt2.hidden_dim, 768);
    assert_eq!(gpt2.num_heads, 12);
    assert_eq!(gpt2.head_dim, 64);
    
    let llama = GAEConfig::llama_7b();
    assert_eq!(llama.hidden_dim, 4096);
    assert_eq!(llama.num_heads, 32);
    assert_eq!(llama.head_dim, 128);
}

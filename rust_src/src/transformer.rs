//! Complete Transformer block with fused GAE operations

use crate::config::GAEConfig;
use crate::mlp::fused_mlp_layernorm;
use crate::layernorm::layernorm;

/// Configuration for a transformer block
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Hidden dimension (embedding size)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// MLP intermediate dimension
    pub mlp_dim: usize,
    /// LayerNorm epsilon
    pub eps: f32,
}

impl From<&GAEConfig> for TransformerConfig {
    fn from(cfg: &GAEConfig) -> Self {
        Self {
            hidden_dim: cfg.hidden_dim,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            mlp_dim: cfg.mlp_dim,
            eps: cfg.ln_eps,
        }
    }
}

/// Complete transformer block with fused operations
pub struct TransformerBlock {
    /// Block configuration
    pub config: TransformerConfig,
    /// Query projection weights
    pub wq: Vec<f32>,
    /// Key projection weights
    pub wk: Vec<f32>,
    /// Value projection weights
    pub wv: Vec<f32>,
    /// Output projection weights
    pub wo: Vec<f32>,
    /// Pre-attention LayerNorm gamma
    pub ln1_gamma: Vec<f32>,
    /// Pre-attention LayerNorm beta
    pub ln1_beta: Vec<f32>,
    /// MLP first layer weights
    pub w1: Vec<f32>,
    /// MLP first layer bias
    pub b1: Vec<f32>,
    /// MLP second layer weights
    pub w2: Vec<f32>,
    /// MLP second layer bias
    pub b2: Vec<f32>,
    /// Post-MLP LayerNorm gamma
    pub ln2_gamma: Vec<f32>,
    /// Post-MLP LayerNorm beta
    pub ln2_beta: Vec<f32>,
}

impl TransformerBlock {
    /// Create with small random initialization (for testing)
    pub fn new_random(config: TransformerConfig) -> Self {
        let h = config.hidden_dim;
        let m = config.mlp_dim;
        Self {
            wq: vec![0.02; h * h],
            wk: vec![0.02; h * h],
            wv: vec![0.02; h * h],
            wo: vec![0.02; h * h],
            ln1_gamma: vec![1.0; h],
            ln1_beta: vec![0.0; h],
            w1: vec![0.02; h * m],
            b1: vec![0.0; m],
            w2: vec![0.02; m * h],
            b2: vec![0.0; h],
            ln2_gamma: vec![1.0; h],
            ln2_beta: vec![0.0; h],
            config,
        }
    }

    /// Forward pass with fused operations
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_dim;
        let mut output = vec![0.0; seq_len * h];
        
        for pos in 0..seq_len {
            let row_start = pos * h;
            let row_end = row_start + h;
            let row = &input[row_start..row_end];
            
            // Pre-attention LayerNorm
            let normed = layernorm(row, &self.ln1_gamma, &self.ln1_beta, cfg.eps);
            
            // Project V (simplified single-position attention for demo)
            let v = matvec(&self.wv, &normed, h, h);
            let attn_out = v;
            
            // Output projection + residual
            let projected = matvec(&self.wo, &attn_out, h, h);
            let residual1: Vec<f32> = projected.iter().zip(row.iter())
                .map(|(&p, &i)| p + i).collect();
            
            // Fused MLP + LayerNorm + Residual
            let final_out = fused_mlp_layernorm(
                &residual1, &residual1,
                &self.w1, &self.b1, &self.w2, &self.b2,
                &self.ln2_gamma, &self.ln2_beta,
                h, cfg.mlp_dim, cfg.eps,
            );
            
            output[row_start..row_end].copy_from_slice(&final_out);
        }
        output
    }
}

/// Matrix-vector multiply: W[m,n] * x[n] -> y[m]
fn matvec(w: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += w[i * n + j] * x[j];
        }
        y[i] = sum;
    }
    y
}

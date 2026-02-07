//! Configuration for GAE

/// Global configuration for the Geodesic Attention Engine
#[derive(Debug, Clone)]
pub struct GAEConfig {
    /// Model hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (hidden_dim / num_heads)
    pub head_dim: usize,
    /// MLP intermediate dimension (typically 4x hidden_dim)
    pub mlp_dim: usize,
    /// LayerNorm epsilon
    pub ln_eps: f32,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl GAEConfig {
    /// Create a new configuration
    pub fn new(hidden_dim: usize, num_heads: usize, mlp_dim: usize, max_seq_len: usize) -> Self {
        assert!(hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads");
        Self {
            hidden_dim,
            num_heads,
            head_dim: hidden_dim / num_heads,
            mlp_dim,
            ln_eps: 1e-5,
            max_seq_len,
        }
    }

    /// GPT-2 Small configuration
    pub fn gpt2_small() -> Self {
        Self::new(768, 12, 3072, 1024)
    }

    /// GPT-2 Medium configuration  
    pub fn gpt2_medium() -> Self {
        Self::new(1024, 16, 4096, 1024)
    }

    /// LLaMA-7B configuration
    pub fn llama_7b() -> Self {
        Self::new(4096, 32, 11008, 4096)
    }
}

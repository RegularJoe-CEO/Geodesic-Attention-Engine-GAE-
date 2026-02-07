//! # Geodesic Attention Engine (GAE)
//! 
//! Physics-principled compute architecture for 75-85% energy reduction.
//! 
//! © 2026 Eric Waller. All Rights Reserved.
//! CONFIDENTIAL — NOT FOR DISTRIBUTION

#![warn(missing_docs)]

pub mod config;
pub mod online_softmax;
pub mod welford;
pub mod activations;
pub mod waller_operator;
pub mod layernorm;
pub mod mlp;
pub mod transformer;

pub use config::GAEConfig;
pub use online_softmax::{OnlineSoftmax, OnlineSoftmaxVec};
pub use welford::WelfordState;
pub use activations::{gelu, silu, relu};
pub use waller_operator::{waller_operator, waller_operator_parallel};
pub use layernorm::{layernorm, layernorm_batched};
pub use mlp::{mlp_block, fused_mlp_layernorm};
pub use transformer::{TransformerBlock, TransformerConfig};

pub mod gpu;

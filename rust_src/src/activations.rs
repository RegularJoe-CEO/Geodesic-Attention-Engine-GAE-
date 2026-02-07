//! Activation functions

/// GELU activation (exact)
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// SiLU / Swish activation
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ReLU activation
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// GELU for a slice (in-place)
pub fn gelu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = gelu(*x);
    }
}

/// SiLU for a slice (in-place)
pub fn silu_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = silu(*x);
    }
}

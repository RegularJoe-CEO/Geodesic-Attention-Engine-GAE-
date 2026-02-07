//! WebGPU backend for GAE - unlocks Metal on Mac

#[cfg(feature = "wgpu")]
mod wgpu_backend;

#[cfg(feature = "wgpu")]
pub use wgpu_backend::*;

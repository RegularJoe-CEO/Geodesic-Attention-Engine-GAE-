//! Waller Operator: Fused Online-Softmax Attention
//! © 2026 Eric Waller. All Rights Reserved.

use crate::online_softmax::OnlineSoftmax;

/// Waller Operator: Single-pass fused attention with O(N) memory
/// Combines QK scoring, online softmax, and value accumulation
pub fn waller_operator(
    q: &[f32],        // [seq_len, head_dim]
    k: &[f32],        // [seq_len, head_dim]  
    v: &[f32],        // [seq_len, head_dim]
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; seq_len * head_dim];
    
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let mut softmax = OnlineSoftmax::new();
        let mut acc = vec![0.0; head_dim];
        
        // Single pass: compute scores, update softmax, accumulate values
        for j in 0..=i {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            
            // QK dot product
            let score: f32 = q_row.iter().zip(k_row.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum::<f32>() * scale;
            
            // Online softmax update with value accumulation
            let old_max = softmax.max;
            softmax.update(score);
            
            // Rescale previous accumulator
            if old_max != f32::NEG_INFINITY {
                let correction = (old_max - softmax.max).exp();
                for a in acc.iter_mut() {
                    *a *= correction;
                }
            }
            
            // Add current value contribution
            let weight = (score - softmax.max).exp();
            for (a, &vi) in acc.iter_mut().zip(v_row.iter()) {
                *a += weight * vi;
            }
        }
        
        // Normalize and store
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for (o, a) in out_row.iter_mut().zip(acc.iter()) {
            *o = a / softmax.sum;
        }
    }
    output
}

/// Parallel Waller Operator using Rayon
#[cfg(feature = "rayon")]
pub fn waller_operator_parallel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    
    let mut output = vec![0.0; seq_len * head_dim];
    
    output.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(i, out_row)| {
            let q_row = &q[i * head_dim..(i + 1) * head_dim];
            let mut softmax = OnlineSoftmax::new();
            let mut acc = vec![0.0; head_dim];
            
            for j in 0..=i {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                
                let score: f32 = q_row.iter().zip(k_row.iter())
                    .map(|(&qi, &ki)| qi * ki)
                    .sum::<f32>() * scale;
                
                let old_max = softmax.max;
                softmax.update(score);
                
                if old_max != f32::NEG_INFINITY {
                    let correction = (old_max - softmax.max).exp();
                    for a in acc.iter_mut() { *a *= correction; }
                }
                
                let weight = (score - softmax.max).exp();
                for (a, &vi) in acc.iter_mut().zip(v_row.iter()) {
                    *a += weight * vi;
                }
            }
            
            for (o, a) in out_row.iter_mut().zip(acc.iter()) {
                *o = a / softmax.sum;
            }
        });
    output
}

#[cfg(not(feature = "rayon"))]
pub fn waller_operator_parallel(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, head_dim: usize, scale: f32,
) -> Vec<f32> {
    waller_operator(q, k, v, seq_len, head_dim, scale)
}

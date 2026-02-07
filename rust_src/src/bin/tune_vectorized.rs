//! Vectorized Waller Operator for M1 Pro
//! Use vec4 loads to maximize memory bandwidth

use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER_VEC4: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    seq_len: u32,
    head_dim_vec4: u32,  // head_dim / 4
    scale: f32,
    _pad: u32,
}

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    let seq_len = params.seq_len;
    let hd4 = params.head_dim_vec4;  // 32 for head_dim=128
    let scale = params.scale;
    
    if (pos >= seq_len) { return; }
    
    var max_score: f32 = -1e38;
    var sum_exp: f32 = 0.0;
    
    // 32 vec4s = 128 floats
    var acc0: vec4<f32> = vec4(0.0); var acc1: vec4<f32> = vec4(0.0);
    var acc2: vec4<f32> = vec4(0.0); var acc3: vec4<f32> = vec4(0.0);
    var acc4: vec4<f32> = vec4(0.0); var acc5: vec4<f32> = vec4(0.0);
    var acc6: vec4<f32> = vec4(0.0); var acc7: vec4<f32> = vec4(0.0);
    var acc8: vec4<f32> = vec4(0.0); var acc9: vec4<f32> = vec4(0.0);
    var acc10: vec4<f32> = vec4(0.0); var acc11: vec4<f32> = vec4(0.0);
    var acc12: vec4<f32> = vec4(0.0); var acc13: vec4<f32> = vec4(0.0);
    var acc14: vec4<f32> = vec4(0.0); var acc15: vec4<f32> = vec4(0.0);
    var acc16: vec4<f32> = vec4(0.0); var acc17: vec4<f32> = vec4(0.0);
    var acc18: vec4<f32> = vec4(0.0); var acc19: vec4<f32> = vec4(0.0);
    var acc20: vec4<f32> = vec4(0.0); var acc21: vec4<f32> = vec4(0.0);
    var acc22: vec4<f32> = vec4(0.0); var acc23: vec4<f32> = vec4(0.0);
    var acc24: vec4<f32> = vec4(0.0); var acc25: vec4<f32> = vec4(0.0);
    var acc26: vec4<f32> = vec4(0.0); var acc27: vec4<f32> = vec4(0.0);
    var acc28: vec4<f32> = vec4(0.0); var acc29: vec4<f32> = vec4(0.0);
    var acc30: vec4<f32> = vec4(0.0); var acc31: vec4<f32> = vec4(0.0);
    
    let q_base = pos * hd4;
    
    for (var j: u32 = 0u; j <= pos; j++) {
        let k_base = j * hd4;
        
        // Vectorized dot product: Q·K
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < hd4; d++) {
            let qv = q[q_base + d];
            let kv = k[k_base + d];
            score += dot(qv, kv);
        }
        score *= scale;
        
        // Online softmax
        let new_max = max(max_score, score);
        let correction = exp(max_score - new_max);
        let weight = exp(score - new_max);
        
        sum_exp = sum_exp * correction + weight;
        
        // Accumulate V with correction
        let v_base = j * hd4;
        acc0 = acc0 * correction + weight * v[v_base + 0u];
        acc1 = acc1 * correction + weight * v[v_base + 1u];
        acc2 = acc2 * correction + weight * v[v_base + 2u];
        acc3 = acc3 * correction + weight * v[v_base + 3u];
        acc4 = acc4 * correction + weight * v[v_base + 4u];
        acc5 = acc5 * correction + weight * v[v_base + 5u];
        acc6 = acc6 * correction + weight * v[v_base + 6u];
        acc7 = acc7 * correction + weight * v[v_base + 7u];
        acc8 = acc8 * correction + weight * v[v_base + 8u];
        acc9 = acc9 * correction + weight * v[v_base + 9u];
        acc10 = acc10 * correction + weight * v[v_base + 10u];
        acc11 = acc11 * correction + weight * v[v_base + 11u];
        acc12 = acc12 * correction + weight * v[v_base + 12u];
        acc13 = acc13 * correction + weight * v[v_base + 13u];
        acc14 = acc14 * correction + weight * v[v_base + 14u];
        acc15 = acc15 * correction + weight * v[v_base + 15u];
        acc16 = acc16 * correction + weight * v[v_base + 16u];
        acc17 = acc17 * correction + weight * v[v_base + 17u];
        acc18 = acc18 * correction + weight * v[v_base + 18u];
        acc19 = acc19 * correction + weight * v[v_base + 19u];
        acc20 = acc20 * correction + weight * v[v_base + 20u];
        acc21 = acc21 * correction + weight * v[v_base + 21u];
        acc22 = acc22 * correction + weight * v[v_base + 22u];
        acc23 = acc23 * correction + weight * v[v_base + 23u];
        acc24 = acc24 * correction + weight * v[v_base + 24u];
        acc25 = acc25 * correction + weight * v[v_base + 25u];
        acc26 = acc26 * correction + weight * v[v_base + 26u];
        acc27 = acc27 * correction + weight * v[v_base + 27u];
        acc28 = acc28 * correction + weight * v[v_base + 28u];
        acc29 = acc29 * correction + weight * v[v_base + 29u];
        acc30 = acc30 * correction + weight * v[v_base + 30u];
        acc31 = acc31 * correction + weight * v[v_base + 31u];
        
        max_score = new_max;
    }
    
    let inv_sum = 1.0 / sum_exp;
    let o_base = pos * hd4;
    output[o_base + 0u] = acc0 * inv_sum;
    output[o_base + 1u] = acc1 * inv_sum;
    output[o_base + 2u] = acc2 * inv_sum;
    output[o_base + 3u] = acc3 * inv_sum;
    output[o_base + 4u] = acc4 * inv_sum;
    output[o_base + 5u] = acc5 * inv_sum;
    output[o_base + 6u] = acc6 * inv_sum;
    output[o_base + 7u] = acc7 * inv_sum;
    output[o_base + 8u] = acc8 * inv_sum;
    output[o_base + 9u] = acc9 * inv_sum;
    output[o_base + 10u] = acc10 * inv_sum;
    output[o_base + 11u] = acc11 * inv_sum;
    output[o_base + 12u] = acc12 * inv_sum;
    output[o_base + 13u] = acc13 * inv_sum;
    output[o_base + 14u] = acc14 * inv_sum;
    output[o_base + 15u] = acc15 * inv_sum;
    output[o_base + 16u] = acc16 * inv_sum;
    output[o_base + 17u] = acc17 * inv_sum;
    output[o_base + 18u] = acc18 * inv_sum;
    output[o_base + 19u] = acc19 * inv_sum;
    output[o_base + 20u] = acc20 * inv_sum;
    output[o_base + 21u] = acc21 * inv_sum;
    output[o_base + 22u] = acc22 * inv_sum;
    output[o_base + 23u] = acc23 * inv_sum;
    output[o_base + 24u] = acc24 * inv_sum;
    output[o_base + 25u] = acc25 * inv_sum;
    output[o_base + 26u] = acc26 * inv_sum;
    output[o_base + 27u] = acc27 * inv_sum;
    output[o_base + 28u] = acc28 * inv_sum;
    output[o_base + 29u] = acc29 * inv_sum;
    output[o_base + 30u] = acc30 * inv_sum;
    output[o_base + 31u] = acc31 * inv_sum;
}
"#;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("         VECTORIZED WALLER OPERATOR — M1 Pro");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default()
    })).unwrap();
    
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vec4_waller"),
        source: wgpu::ShaderSource::Wgsl(SHADER_VEC4.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vec4_pipeline"), layout: Some(&pipeline_layout), module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let configs: [(usize, &str); 4] = [
        (2048, "2K"),
        (4096, "4K"),
        (8192, "8K"),
        (16384, "16K"),
    ];

    println!("{:<8} {:>12} {:>12} {:>14}", "Context", "Time (ms)", "GFLOPS", "vs Previous");
    println!("{}", "─".repeat(50));

    // Previous best from gpu_energy_benchmark
    let previous_gflops = [292.9, 540.1, 905.6, 0.0]; // No 16K data

    for (idx, &(seq_len, name)) in configs.iter().enumerate() {
        let head_dim: usize = 128;
        let head_dim_vec4 = head_dim / 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.01).cos() * 0.1).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.02).sin() * 0.1).collect();

        let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q"), contents: bytemuck::cast_slice(&q), usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("k"), contents: bytemuck::cast_slice(&k), usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v"), contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"), size: (seq_len * head_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let params: [u32; 4] = [seq_len as u32, head_dim_vec4 as u32, scale.to_bits(), 0];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let workgroups = (seq_len as u32 + 15) / 16;

        // Warmup
        for _ in 0..3 {
            let mut encoder = device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]);
              pass.dispatch_workgroups(workgroups, 1, 1); }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        let iterations = match seq_len { 2048 => 20, 4096 => 10, _ => 5 };
        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = device.create_command_encoder(&Default::default());
            { let mut pass = encoder.begin_compute_pass(&Default::default());
              pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]);
              pass.dispatch_workgroups(workgroups, 1, 1); }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        let flops = (seq_len as f64) * (seq_len as f64 / 2.0) * (4.0 * head_dim as f64 + 6.0);
        let gflops = flops / (time_ms * 1e6);
        
        let vs_prev = if previous_gflops[idx] > 0.0 { 
            format!("{:.2}×", gflops / previous_gflops[idx]) 
        } else { 
            "N/A".to_string() 
        };

        println!("{:<8} {:>12.2} {:>12.1} {:>14}", name, time_ms, gflops, vs_prev);
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
}

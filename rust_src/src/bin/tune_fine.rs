//! Fine-grained tuning for M1 Pro
//! Test small workgroups + tiling strategies

use std::time::Instant;
use wgpu::util::DeviceExt;

fn create_shader(workgroup_size: u32, tile_k: u32) -> String {
    format!(r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {{
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    _pad: u32,
}}

const TILE_K: u32 = {tile_k}u;

var<workgroup> k_tile: array<f32, {tile_mem}>;  // TILE_K * 128
var<workgroup> v_tile: array<f32, {tile_mem}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {{
    let pos = gid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let scale = params.scale;
    
    if (pos >= seq_len) {{ return; }}
    
    var max_score: f32 = -1e38;
    var sum_exp: f32 = 0.0;
    var acc = array<f32, 128>();
    
    for (var d: u32 = 0u; d < 128u; d++) {{
        acc[d] = 0.0;
    }}
    
    // Process in tiles for better cache usage
    let num_tiles = (pos + TILE_K) / TILE_K;
    
    for (var tile: u32 = 0u; tile < num_tiles; tile++) {{
        let tile_start = tile * TILE_K;
        let tile_end = min(tile_start + TILE_K, pos + 1u);
        
        for (var j: u32 = tile_start; j < tile_end; j++) {{
            var score: f32 = 0.0;
            for (var d: u32 = 0u; d < 128u; d++) {{
                score += q[pos * 128u + d] * k[j * 128u + d];
            }}
            score *= scale;
            
            let new_max = max(max_score, score);
            let correction = exp(max_score - new_max);
            let weight = exp(score - new_max);
            
            sum_exp = sum_exp * correction + weight;
            for (var d: u32 = 0u; d < 128u; d++) {{
                acc[d] = acc[d] * correction + weight * v[j * 128u + d];
            }}
            max_score = new_max;
        }}
    }}
    
    let inv_sum = 1.0 / sum_exp;
    for (var d: u32 = 0u; d < 128u; d++) {{
        output[pos * 128u + d] = acc[d] * inv_sum;
    }}
}}
"#, 
    workgroup_size = workgroup_size,
    tile_k = tile_k,
    tile_mem = tile_k * 128
    )
}

fn create_pipeline(device: &wgpu::Device, workgroup_size: u32, tile_k: u32) -> wgpu::ComputePipeline {
    let shader_source = create_shader(workgroup_size, tile_k);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tuned"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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
    
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tuned_pipeline"), layout: Some(&pipeline_layout), module: &shader,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    })
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("         M1 PRO FINE TUNING — Workgroup × Tile Size");
    println!("═══════════════════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default()
    })).unwrap();
    
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_compute_workgroup_size_x: 256,
                max_compute_invocations_per_workgroup: 256,
                ..Default::default()
            },
            ..Default::default()
        },
        None,
    )).unwrap();

    let seq_len: usize = 4096;
    let head_dim: usize = 128;
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
    let params: [u32; 4] = [seq_len as u32, head_dim as u32, scale.to_bits(), 0];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM,
    });

    let workgroup_sizes = [16, 32, 64, 128, 256];
    let tile_sizes = [16, 32, 64, 128];
    
    println!("{:<8} {:>8} {:>12} {:>12}", "WG", "Tile", "Time (ms)", "GFLOPS");
    println!("{}", "─".repeat(44));
    
    let mut best_gflops = 0.0f64;
    let mut best_config = (0u32, 0u32);
    
    for &wg_size in &workgroup_sizes {
        for &tile_k in &tile_sizes {
            let pipeline = create_pipeline(&device, wg_size, tile_k);
            
            let bind_group_layout = pipeline.get_bind_group_layout(0);
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
            
            let workgroups = (seq_len as u32 + wg_size - 1) / wg_size;
            
            // Warmup
            for _ in 0..3 {
                let mut encoder = device.create_command_encoder(&Default::default());
                { let mut pass = encoder.begin_compute_pass(&Default::default());
                  pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]);
                  pass.dispatch_workgroups(workgroups, 1, 1); }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
            }
            
            // Benchmark
            let iterations = 20;
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
            
            if gflops > best_gflops {
                best_gflops = gflops;
                best_config = (wg_size, tile_k);
            }
            
            println!("{:<8} {:>8} {:>12.2} {:>12.1}", wg_size, tile_k, time_ms, gflops);
        }
    }
    
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("  BEST: Workgroup={}, Tile={} → {:.1} GFLOPS", best_config.0, best_config.1, best_gflops);
    println!("═══════════════════════════════════════════════════════════════════════");
}

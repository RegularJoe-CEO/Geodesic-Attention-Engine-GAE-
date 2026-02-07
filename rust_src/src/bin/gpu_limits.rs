//! Query M1 Pro GPU limits and capabilities

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("              M1 PRO GPU CAPABILITIES");
    println!("═══════════════════════════════════════════════════════════════\n");

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).expect("No adapter");

    let info = adapter.get_info();
    println!("Device: {} ({:?})", info.name, info.backend);
    println!("");

    let limits = adapter.limits();
    println!("COMPUTE LIMITS:");
    println!("  Max workgroup size X:        {}", limits.max_compute_workgroup_size_x);
    println!("  Max workgroup size Y:        {}", limits.max_compute_workgroup_size_y);
    println!("  Max workgroup size Z:        {}", limits.max_compute_workgroup_size_z);
    println!("  Max workgroup invocations:   {}", limits.max_compute_workgroups_per_dimension);
    println!("  Max invocations per workgrp: {}", limits.max_compute_invocations_per_workgroup);
    println!("");
    
    println!("MEMORY LIMITS:");
    println!("  Max buffer size:             {} MB", limits.max_buffer_size / 1_000_000);
    println!("  Max storage buffer binding:  {} MB", limits.max_storage_buffer_binding_size / 1_000_000);
    println!("  Max uniform buffer binding:  {} KB", limits.max_uniform_buffer_binding_size / 1_000);
    println!("");

    println!("BINDING LIMITS:");
    println!("  Max bind groups:             {}", limits.max_bind_groups);
    println!("  Max bindings per group:      {}", limits.max_bindings_per_bind_group);
    println!("  Max storage buffers/stage:   {}", limits.max_storage_buffers_per_shader_stage);
    println!("");

    let features = adapter.features();
    println!("KEY FEATURES:");
    println!("  Timestamp queries:           {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
    println!("  Shader f16:                  {}", features.contains(wgpu::Features::SHADER_F16));
    println!("  Subgroups:                   {}", features.contains(wgpu::Features::SUBGROUP));
    
    println!("\n═══════════════════════════════════════════════════════════════");
}

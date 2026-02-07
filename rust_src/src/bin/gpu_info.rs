fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No adapter found");

    let info = adapter.get_info();
    println!("=== GPU INFO ===");
    println!("Name:    {}", info.name);
    println!("Vendor:  {:?}", info.vendor);
    println!("Backend: {:?}", info.backend);
    println!("Type:    {:?}", info.device_type);
    println!("Driver:  {}", info.driver);
}

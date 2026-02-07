//! Welford's Online Algorithm for streaming mean/variance

/// Welford accumulator for numerically stable streaming statistics
#[derive(Debug, Clone)]
pub struct WelfordState {
    /// Running mean
    pub mean: f32,
    /// Running M2 (sum of squared differences)
    pub m2: f32,
    /// Sample count
    pub count: u32,
}

impl Default for WelfordState {
    fn default() -> Self {
        Self::new()
    }
}

impl WelfordState {
    /// Create new accumulator
    pub fn new() -> Self {
        Self { mean: 0.0, m2: 0.0, count: 0 }
    }

    /// Update with new sample in O(1)
    pub fn update(&mut self, x: f32) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get population variance
    pub fn variance(&self) -> f32 {
        if self.count < 2 { 0.0 } else { self.m2 / self.count as f32 }
    }

    /// Get standard deviation with epsilon for stability
    pub fn std(&self, eps: f32) -> f32 {
        (self.variance() + eps).sqrt()
    }

    /// Merge two Welford states (parallel reduction)
    pub fn merge(a: &WelfordState, b: &WelfordState) -> WelfordState {
        let count = a.count + b.count;
        if count == 0 {
            return WelfordState::new();
        }
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * b.count as f32 / count as f32;
        let m2 = a.m2 + b.m2 + delta * delta * a.count as f32 * b.count as f32 / count as f32;
        WelfordState { mean, m2, count }
    }
}

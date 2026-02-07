//! Online Softmax: Numerically stable single-pass softmax
//! Based on Milakov & Gimelshein 2018

/// Single-pass online softmax accumulator
pub struct OnlineSoftmax {
    /// Running maximum score
    pub max: f32,
    /// Running sum of exp(score - max)
    pub sum: f32,
}

impl Default for OnlineSoftmax {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineSoftmax {
    /// Create new accumulator
    pub fn new() -> Self {
        Self {
            max: f32::NEG_INFINITY,
            sum: 0.0,
        }
    }

    /// Update with new score in O(1)
    pub fn update(&mut self, score: f32) {
        if score > self.max {
            self.sum = self.sum * (self.max - score).exp() + 1.0;
            self.max = score;
        } else {
            self.sum += (score - self.max).exp();
        }
    }

    /// Get probability for a score
    pub fn probability(&self, score: f32) -> f32 {
        (score - self.max).exp() / self.sum
    }
}

/// Multi-head online softmax
pub struct OnlineSoftmaxVec {
    /// Per-head softmax states
    pub states: Vec<OnlineSoftmax>,
}

impl OnlineSoftmaxVec {
    /// Create for n_heads
    pub fn new(n_heads: usize) -> Self {
        Self {
            states: (0..n_heads).map(|_| OnlineSoftmax::new()).collect(),
        }
    }

    /// Update specific head
    pub fn update(&mut self, head: usize, score: f32) {
        self.states[head].update(score);
    }

    /// Reset all heads
    pub fn reset(&mut self) {
        for state in &mut self.states {
            *state = OnlineSoftmax::new();
        }
    }
}

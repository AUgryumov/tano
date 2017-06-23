/// Enum for layer optimization
pub enum LayerOptimizationTypes {
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Trait which allows you to optimize layer
pub trait LayerOptimizer {
    fn optimize(&mut self, optimizer: LayerOptimizationTypes);
}
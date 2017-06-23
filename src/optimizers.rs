// TODO WRITE OPTIMIZATION METHODS
#![allow(dead_code)]

/// Enum in which you can choose optimization method
pub enum OptimizationTypes {
    /// Feed forward optimization method.
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Enum for layer optimization
pub(crate) enum LayerOptimizationTypes<'a> {
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Enum for neuron optimization
pub(crate) enum NeuronOptimizationTypes {
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Trait which allows you to optimize network
pub trait Optimizer {
    fn optimize(&mut self, optimizer: OptimizationTypes);
}

/// Trait which allows you to optimize layer
pub(crate) trait LayerOptimizer {
    fn optimize(&mut self, optimizer: LayerOptimizationTypes);
}

/// Trait which allows you to optimize neuron
pub(crate) trait NeuronOptimizer {
    fn optimize(&mut self, optimizer: NeuronOptimizationTypes);
}
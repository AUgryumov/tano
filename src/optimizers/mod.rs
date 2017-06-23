#![allow(dead_code)]

/// Enum in which you can choose optimization method
pub enum OptimizationTypes {
    /// Feed forward optimization method.
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Trait which allows you to optimize network
pub trait Optimizer {
    fn optimize(&mut self, optimizer: OptimizationTypes);
}

pub mod layer_optimizers;
pub mod neuron_optimizers;

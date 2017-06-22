// TODO WRITE OPTIMIZATION METHODS
#![allow(dead_code)]

pub enum OptimizationTypes {
    FeedForward(Vec<f64>, Vec<f64>, f64) // input, expected and learning rate
}

pub(crate) enum LayerOptimizationTypes {
    FeedForward(Vec<f64>, Vec<f64>, f64) // input, expected and learning rate
}

pub(crate) enum NeuronOptimizationTypes {
    FeedForward(Vec<f64>, Vec<f64>, f64) // input, expected and learning rate
}

pub trait Optimizer {
    fn optimize(&mut self, optimizer: OptimizationTypes);
}

pub(crate) trait LayerOptimizer {
    fn optimize(&mut self, optimizer: LayerOptimizationTypes);
}

pub(crate) trait NeuronOptimizer {
    fn optimize(&mut self, optimizer: NeuronOptimizationTypes);
}
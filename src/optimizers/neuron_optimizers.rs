/// Enum for neuron optimization
pub enum NeuronOptimizationTypes {
    FeedForward{input: Vec<f64>, expected: Vec<f64>, learning_rate: f64}
}

/// Trait which allows you to optimize neuron
pub trait NeuronOptimizer {
    fn optimize(&mut self, optimizer: NeuronOptimizationTypes);
}
use super::neurons::{Neuron, UsualNeuron};
use super::optimizers::layer_optimizers::{LayerOptimizer, LayerOptimizationModes};
use super::utils::activation::Activation;

/// A lot of this traits forms a network
pub trait Layer: LayerOptimizer {
    /// Creates new Layer.
    fn new(neurons_count: usize) -> Self where Self: Sized;

    /// Sets relation count (count of links with other layer)
    /// More often this is count of neurons in previous layer
    fn set_relations_count(&mut self, relations_count: usize);

    /// Runs a layer.
    /// Mutable pointer to the `self` is necessary for recurrent networks
    fn run(&mut self, input: Vec<f64>, activation: Activation) -> Vec<f64>;

    /// Returns a pointer to neurons of the layer
    fn get_neurons(&self) -> &Vec<Box<Neuron>>;

    /// Returns a mutable pointer to neurons of the layer
    fn get_mut_neurons(&mut self) -> &mut Vec<Box<Neuron>>;
}

/// Simple layer structure
pub struct UsualLayer {
    neurons: Vec<Box<Neuron>>
}

impl Layer for UsualLayer {
    fn new(neurons_count: usize) -> Self where Self: Sized {
        let mut neurons: Vec<Box<Neuron>> = Vec::with_capacity(neurons_count);
        for _ in 0..neurons_count {
            neurons.push(Box::new(UsualNeuron::new(0)));
        }
        UsualLayer {
            neurons
        }
    }

    fn set_relations_count(&mut self, relations_count: usize) {
        let mut new_neurons: Vec<Box<Neuron>> = Vec::with_capacity(self.neurons.len());
        for _ in 0..self.neurons.len() {
            new_neurons.push(Box::new(UsualNeuron::new(relations_count)));
        }
        self.neurons = new_neurons;
    }

    fn run(&mut self, input: Vec<f64>, activation: Activation) -> Vec<f64> {
        let mut output = Vec::with_capacity(self.neurons.len());
        for neuron in &mut self.neurons {
            output.push(neuron.run(&input, activation));
        }
        output
    }

    fn get_neurons(&self) -> &Vec<Box<Neuron>> {
        &self.neurons
    }

    fn get_mut_neurons(&mut self) -> &mut Vec<Box<Neuron>> {
        &mut self.neurons
    }
}

impl LayerOptimizer for UsualLayer {
    // TODO IMPLEMENT
    fn optimize(&mut self, optimizer: LayerOptimizationModes) {
        unimplemented!()
    }
}
extern crate rand;

use self::rand::Rng;

use super::optimizers::neuron_optimizers::{NeuronOptimizer, NeuronOptimizationTypes};
use super::utils::activation::Activation;

/// A lot of this traits forms a layer
pub trait Neuron: NeuronOptimizer {
    /// Creates new neuron
    fn new(relation_count: usize) -> Self where Self: Sized;

    /// Performs input
    fn run(&mut self, input: &Vec<f64>, activation: Activation) -> f64;

    /// Returns a pointer to weights of the neuron
    fn get_weights(&self) -> &Vec<f64>;

    /// Returns a mutable pointer to weights of the neuron
    fn get_mut_weights(&mut self) -> &mut Vec<f64>;
}

/// Simple neuron structure
pub(crate) struct UsualNeuron {
    weights: Vec<f64>
}

impl Neuron for UsualNeuron {
    fn new(relation_count: usize) -> Self where Self: Sized {
        // Weights generation
        let mut weights: Vec<f64> = Vec::with_capacity(relation_count);
        for _ in 0..relation_count {
            weights.push(rand::thread_rng().gen());
        }
        UsualNeuron {
            weights
        }
    }

    fn run(&mut self, input: &Vec<f64>, activation: Activation) -> f64 {
        // TODO DELETE
        if input.len() != self.weights.len() {
            panic!("dimension of input and weights vectors do not match");
        }

        // Sums and runs activation
        activation(input.iter().zip(self.weights.iter()).fold(0f64, |s, (i, w)| s + w * i))
    }

    fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    fn get_mut_weights(&mut self) -> &mut Vec<f64> {
        &mut self.weights
    }
}

impl NeuronOptimizer for UsualNeuron {
    fn optimize(&mut self, optimizer: NeuronOptimizationTypes) {
        unimplemented!()
    }
}

/// Recurrent neuron structure
pub(crate) struct RecurrentNeuron {
    neuron: UsualNeuron,
    recurrent_connection: f64
}

impl Neuron for RecurrentNeuron {
    fn new(relation_count: usize) -> Self where Self: Sized {
        // Weights generation (+1 for recurrent connection)
        RecurrentNeuron {
            neuron: UsualNeuron::new(relation_count + 1),
            recurrent_connection: 0.0
        }
    }

    fn run(&mut self, input: &Vec<f64>, activation: Activation) -> f64 {
        // Adds recurrent connection value
        let mut input_with_recurrent_connection = input.clone();
        input_with_recurrent_connection.push(self.recurrent_connection);

        let result = self.neuron.run(&input_with_recurrent_connection, activation);
        self.recurrent_connection = result;
        result
    }

    fn get_weights(&self) -> &Vec<f64> {
        self.neuron.get_weights()
    }

    fn get_mut_weights(&mut self) -> &mut Vec<f64> {
        self.neuron.get_mut_weights()
    }
}

impl NeuronOptimizer for RecurrentNeuron {
    fn optimize(&mut self, optimizer: NeuronOptimizationTypes) {
        unimplemented!()
    }
}

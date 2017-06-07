extern crate rand;

use self::rand::Rng;
use super::utils::sigmoid;

// Any struct that implements Layer must realise the functions bellow.
// You can see a sample of implementation of this trait in UsualLayer implementations
pub trait Layer {
    // Creates a new layer. Returns Self (Layer implementation)
    fn new(neuron_count: usize, connections_count: usize) -> Self where Self: Sized;
    // Runs layer. Input: vector of output of a last layer. Returns vector of neuron output
    fn run(&self, input: &Vec<f64>) -> Vec<f64>;
    // Trains layer. `errors`: vector of error for each neuron. `input`: vector of output of a last layer.
    // learning_rate: learning rate for network. Returns vector of new errors
    fn train(&mut self, errors: Vec<f64>, input: Vec<f64>, learning_rate: f64) -> Vec<f64>;
}

// Base neural network layer
pub struct UsualLayer {
    // Neuron list. The neuron consist only of weights
    neurons: Vec<Vec<f64>>,
    // Count of connections of each neuron
    connections_count: usize
}

impl Layer for UsualLayer {
    fn new(neuron_count: usize, connections_count: usize) -> Self {
        let mut rng = rand::thread_rng(); // Range for weight randomize
        let mut neurons: Vec<Vec<f64>> = Vec::with_capacity(neuron_count); // Neuron vec

        // Neurons creating
        for _ in 0..neuron_count {
            // Weights creating
            let mut weights = Vec::new();
            for _ in 0..connections_count {
                weights.push(rng.gen_range(-0.5_f64, 0.5_f64)) // Generates weight from -0.5 to 0.5
            }
            neurons.push(weights);
        }

        UsualLayer {
            neurons,
            connections_count
        }
    }

    fn run(&self, input: &Vec<f64>) -> Vec<f64> {
        // Output vector
        let mut out: Vec<f64> = Vec::new();

        for weights in &self.neurons {
            let mut transferred = 0.;
            
            // Calculating transfer output
            for (input, weight) in input.iter().zip(weights.iter()) {
                transferred += input * weight;
            }

            out.push(sigmoid(transferred));
        }

        out
    }

    // TODO CHECK FORMULAS
    fn train(&mut self, errors: Vec<f64>, input: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        // Error value initializing
        let mut next_layer_errors: Vec<f64> = Vec::with_capacity(self.connections_count);
        for _ in 0..self.connections_count { next_layer_errors.push(0.) }

        for (weights, error) in self.neurons.iter_mut().zip(errors.iter()) {
            // Calculating sigmoid of error
            let sigmoid = sigmoid(*error);

            // Calculating weights delta
            let weights_delta = error * sigmoid * (1. - sigmoid);

            for ((weight, input), error) in weights.iter_mut().zip(input.iter()).zip(next_layer_errors.iter_mut()) {
                // Calculating error for next layer
                // TODO CHECK FORMULE
                *error += *weight * weights_delta;

                // Calculating weight update
                *weight += input * weights_delta * learning_rate;
            }
        }

        next_layer_errors
    }
}

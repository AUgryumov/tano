use std::fmt;

/// Any struct that implements Layer must realise the functions of this trait according to the comments.
/// You can see a sample of implementation of this trait in UsualLayer implementations
pub trait Layer: fmt::Debug {
    /// Creates a new `Layer`
    fn new(neuron_count: usize) -> Self where Self: Sized;

    /// Returns a pointer to neurons
    fn get_weights(&self) -> &Vec<Vec<f64>>;

    /// Returns a mutable pointer to neurons
    fn get_mut_weights(&mut self) -> &mut Vec<Vec<f64>>;

    /// Runs layer. `input`: vector of output of a last layer. Returns vector of neuron output
    fn run(&self, input: &Vec<f64>, activation: &fn(x: f64) -> f64) -> Vec<f64>;
}

/// Base neural network layer
#[derive(Debug)]
pub struct UsualLayer {
    // Weights list. The neuron consist only of weights
    weights: Vec<Vec<f64>>,
}

impl Layer for UsualLayer {
    fn new(neuron_count: usize) -> Self {
        let mut weights: Vec<Vec<f64>> = Vec::with_capacity(neuron_count); // Neuron vec

        // Neurons creating
        for _ in 0..neuron_count {
            weights.push(Vec::new());
        }

        UsualLayer {
            weights,
        }
    }

    fn get_weights(&self)-> &Vec<Vec<f64>> { &self.weights }
    fn get_mut_weights(&mut self)-> &mut Vec<Vec<f64>> { &mut self.weights }

    fn run(&self, input: &Vec<f64>, activation: &fn(x: f64) -> f64) -> Vec<f64> {
        // Output vector
        let mut out: Vec<f64> = Vec::new();

        for weights in &self.weights {
            let mut transferred = 0.;

            // Calculating transfer output
            for (input, weight) in input.iter().zip(weights.iter()) {
                transferred += input * weight;
            }

            out.push(activation(transferred));
        }

        out
    }
}
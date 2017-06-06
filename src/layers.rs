extern crate rand;

use self::rand::Rng;
use super::utils::sigmoid;

pub trait Layer {
    fn new(neuron_count: usize, connections_count: usize) -> Self where Self: Sized; //Returns self
    fn run(&self, input: Vec<f64>) -> Vec<f64>;
    fn learn(&mut self, errors: Vec<f64>, input: Vec<f64>, learning_rate: f64) -> Vec<f64>; //Returns errors
}

pub struct UsualLayer {
    neurons: Vec<Vec<f64>>,
    connections_count: usize
}

impl Layer for UsualLayer {
    fn new(neuron_count: usize, connections_count: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut neurons: Vec<Vec<f64>> = Vec::with_capacity(neuron_count);

        for _ in 0..neuron_count {
            let mut weights = Vec::new();
            for _ in 0..connections_count {
                weights.push(rng.gen_range(-0.5_f64, 0.5_f64))
            }
            neurons.push(weights);
        }

        UsualLayer {
            neurons,
            connections_count
        }
    }

    //TODO CHECK BUGS
    fn run(&self, input: Vec<f64>) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();

        for weights in &self.neurons {
            let mut sum = 0.;
            for (input, weight) in input.iter().zip(weights.iter()) {
                sum += input * weight;
            }
            out.push(sigmoid(sum));
        }

        out
    }

    //TODO CHECK BUGS
    fn learn(&mut self, errors: Vec<f64>, input: Vec<f64>, learning_rate: f64) -> Vec<f64> {
        let mut next_layer_errors: Vec<f64> = Vec::with_capacity(self.connections_count);
        for _ in 0..self.connections_count { next_layer_errors.push(0.) }
        for (weights, error) in self.neurons.iter_mut().zip(errors.iter()) {
            let sigmoid = sigmoid(*error); //Optimization
            let weights_delta = error * sigmoid * (1. - sigmoid);
            println!("SIGMOID: {}; INPUT: {:?}; WEIGHTS_DELTA: {}", sigmoid, input, weights_delta);

            for ((weight, input), error) in weights.iter_mut().zip(input.iter()).zip(next_layer_errors.iter_mut()) {
                *error = *weight * weights_delta;
                println!("FUUUCK: {}; WEIGHT: {}", input * weights_delta * learning_rate, weight);
                *weight += input * weights_delta * learning_rate;
                println!("NEW WEIGHT: {}", weight);
            }
        }
        next_layer_errors
    }
}

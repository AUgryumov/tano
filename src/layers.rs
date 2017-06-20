use super::neurons::{Neuron, UsualNeuron};
use super::utils::Activation;

pub trait Layer {
    fn new(neurons_count: usize) -> Self where Self: Sized;
    fn set_relations_count(&mut self, relations_count: usize);
    fn run(&mut self, input: Vec<f64>, activation: Activation) -> Vec<f64>;
    fn get_neurons(&self) -> &Vec<Box<Neuron>>;
    fn get_mut_neurons(&mut self) -> &mut Vec<Box<Neuron>>;
}

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
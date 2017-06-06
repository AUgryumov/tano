// TODO WRITE TRAINING
// TODO WRITE NETWORK TRAIT
// TODO WRITE NEURON TRAIT
// TODO DOCUMENT NETWORK struct
// TODO DOCUMENT LAYER trait
// TODO WRITE EXAMPLES

// FIXME 1 SAMPLE (1 NEURON, 1 LAYER, NO NETWORK)

pub mod layers;
mod utils;
mod tests;

use layers::Layer;
use layers::UsualLayer;

pub struct Network {
    layers: Vec<Box<Layer>>,
    input_size: usize,
}

impl Network {
    pub fn new(layers_sizes: &[usize]) -> Self {
        if layers_sizes.len() < 2 {
            panic!("Must be at least 2 layers")
        }
        let mut layers: Vec<Box<Layer>> = Vec::with_capacity(layers_sizes.len());
        let mut it = layers_sizes.iter();
        let mut prev_layer_size = *it.next().unwrap();
        let input_size = prev_layer_size.clone();
        for size in it {
            layers.push(Box::new(UsualLayer::new(*size, prev_layer_size)));
            prev_layer_size = *size;
        }

        Network { layers, input_size }
    }

    pub fn run(&self, input: Vec<f64>) -> Vec<f64> {
        if self.input_size != input.len() {
            panic!("Bad input length");
        }

        let mut prev_out = input;
        for layer in &self.layers {
            prev_out = (layer).run(prev_out);
        }

        prev_out
    }
}
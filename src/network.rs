extern crate rand;

use self::rand::Rng;

use layer::Layer;
use layer::UsualLayer;

/// Main `Network` struct
#[derive(Debug)]
pub struct Network {
    // Layers vector
    pub layers: Vec<Box<Layer>>,
    // Size of the first layer (input size)
    input_size: usize,
    // Activation function (sigmoid on default)
    activation: fn(x: f64) -> f64,
}

impl Network {
    /// Runs `Network`
    pub fn run(&self, input: &Vec<f64>) -> Vec<f64> {
        if self.input_size != input.len() {
            panic!("bad input length");
        }

        let mut prev_out = input.clone();
        for layer in &self.layers {
            prev_out = layer.run(&prev_out, &self.activation);
        }

        prev_out
    }
}

/// Struct implements the Builder pattern
pub struct NetworkBuilder {
    // Layers of the new network
    layers: Vec<Box<Layer>>,
    // Activation function
    activation: fn(x: f64) -> f64,
    // Input neuron count
    input_size: usize,
    // Output neuron count
    output_size: usize,
}

impl NetworkBuilder {
    /// Returns default `NetworkBuilder` with input_size and output_size.
    /// This network doesn't contains any layers (except for the output)
    pub fn new(input_size: usize, output_size: usize) -> Self {
        NetworkBuilder {
            layers: Vec::new(),
            activation: super::math::sigmoid,
            input_size,
            output_size,
        }
    }

    /// Adds new `Layer` to the layers vector
    pub fn layer(mut self, layer: Box<Layer>) -> Self {
        let mut layer = layer;

        // Index in which will be located new layer (in layers vector)
        let index = self.layers.len();

        // Checks if layer is first
        let connections_count = if index > 0 {
            self.layers.get_mut(index - 1).unwrap().get_weights().len() // Previous layer length
        } else {
            self.input_size // Input size for the network
        };

        let mut rng = rand::thread_rng(); // Range for weight randomize

        // Neurons creating
        for neuron in layer.get_mut_weights() {
            // Weights creating
            let mut weights = Vec::new();
            for _ in 0..connections_count {
                weights.push(rng.gen_range(-0.5_f64, 0.5_f64)) // Generates weight from -0.5 to 0.5
            }
            *neuron = weights;
        }

        // Inserting new layer in the layers vector
        self.layers.insert(index, layer);
        self
    }

    // Sets activation function
    pub fn activation(mut self, activation: fn(x: f64) -> f64) -> Self {
        self.activation = activation;
        self
    }

    /// Makes `Network` from `NetworkBuilder`
    pub fn finalize(mut self) -> Network {
        let output_size = self.output_size;
        self = self.layer(Box::new(UsualLayer::new(output_size)));

        Network {
            layers: self.layers,
            input_size: self.input_size,
            activation: self.activation
        }
    }
}
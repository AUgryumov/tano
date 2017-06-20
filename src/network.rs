use super::layers::{Layer, UsualLayer};
use super::utils::{Activation, sigmoid};

pub struct Network {
    layers: Vec<Box<Layer>>,
    activation: Activation,
    input_size: usize
}

impl Network {
    pub fn go(&mut self, input: Vec<f64>) -> Vec<f64> {
        if input.len() != self.input_size {
            panic!("bad input length");
        }

        let mut output = input;
        for layer in &mut self.layers {
            output = layer.run(output, self.activation)
        }
        output
    }
}

pub struct NetworkBuilder {
    layers: Vec<Box<Layer>>,
    activation: Activation,
    input_size: usize,
    output_size: usize,
}

impl NetworkBuilder {
    pub fn new(input_size: usize, output_size: usize) -> NetworkBuilder {
        NetworkBuilder {
            layers: Vec::new(),
            activation: sigmoid,
            input_size,
            output_size
        }
    }

    pub fn layer(mut self, layer: Box<Layer>) -> Self {
        let mut new_layer = layer;
        new_layer.set_relations_count(match self.layers.last() {
            Some(T) => T.get_neurons().len(),
            None => self.input_size
        });

        self.layers.push(new_layer);
        self
    }

    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    pub fn finalize(mut self) -> Network {
        let output = self.output_size;
        self = self.layer(Box::new(UsualLayer::new(output)));
        Network {
            layers: self.layers,
            activation: self.activation,
            input_size: self.input_size
        }
    }
}
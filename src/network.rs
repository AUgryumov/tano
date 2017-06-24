use super::layers::{Layer, UsualLayer};
use super::utils::activation::{Activation, sigmoid};
use super::optimizers::{OptimizationModes, Optimizer};

/// A structure that allows you to run and optimize the network
pub struct Network {
    layers: Vec<Box<Layer>>,
    activation: Activation,
    input_size: usize
}

impl Network {
    /// Runs the network
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

impl Optimizer for Network {
    fn optimize(&mut self, optimizer: OptimizationModes) {
        match optimizer {
            OptimizationModes::FeedForward{input: _, expected: _, learning_rate: _} => {
                panic!("feed forward optimization is in development");
            }
        }
    }
}

/// A structure that was made to simplify `Network` creation process
pub struct NetworkBuilder {
    layers: Vec<Box<Layer>>,
    activation: Activation,
    input_size: usize,
}

impl NetworkBuilder {
    /// Creates a new `NetworkBuilder`
    pub fn new(input_size: usize) -> NetworkBuilder {
        NetworkBuilder {
            layers: Vec::new(),
            activation: sigmoid,
            input_size,
        }
    }

    /// Adds a new layer into network
    pub fn layer(mut self, layer: Box<Layer>) -> Self {
        let mut layer = layer;
        layer.set_relations_count(match self.layers.last() {
            Some(l) => l.get_neurons().len(),
            None => self.input_size
        });

        self.layers.push(layer);
        self
    }

    /// Sets network activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Finalizes network creation process
    pub fn finalize(self) -> Network {
        Network {
            layers: self.layers,
            activation: self.activation,
            input_size: self.input_size
        }
    }
}
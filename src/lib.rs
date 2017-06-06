//! An easy to use neural network library written in Rust.
//!
//! # Description
//! tano is a [feedforward neural network ](http://en.wikipedia.org/wiki/Feedforward_neural_network)
//! library. The library
//! generates fully connected multi-layer artificial neural networks that
//! are trained via [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
//! Networks are trained using an incremental training mode.

pub mod trainer;
mod utils;

extern crate rand;
extern crate rustc_serialize;

use rustc_serialize::json;
use rand::Rng;

/// Neural network
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct Network {
    layers: Vec<Vec<Vec<f64>>>,
    num_inputs: u32,
}

impl Network {

    /// Each number in the `layers_sizes` parameter specifies a
    /// layer in the network. The number itself is the number of nodes in that
    /// layer. The first number is the input layer, the last
    /// number is the output layer, and all numbers between the first and
    /// last are hidden layers. There must be at least two layers in the network.
    pub fn new(layers_sizes: &[u32]) -> Network {
        let mut rng = rand::thread_rng();

        if layers_sizes.len() < 2 {
            panic!("must have at least two layers");
        }

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            }
        }


        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();
        // get the first layer size
        let first_layer_size = *it.next().unwrap();

        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_size {
                let mut node: Vec<f64> = Vec::new();
                for _ in 0..prev_layer_size+1 {
                    let random_weight: f64 = rng.gen_range(-0.5f64, 0.5f64);
                    node.push(random_weight);
                }
                node.shrink_to_fit();
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        Network { layers: layers, num_inputs: first_layer_size }
    }

    /// Runs the network on an input and returns a vector of the results.
    /// The number of `f64`s in the input must be the same
    /// as the number of input nodes in the network. The length of the results
    /// vector will be the number of nodes in the output layer of the network.
    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() as u32 != self.num_inputs {
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

    /// Encodes the network as a JSON string.
    pub fn to_json(&self) -> String {
        json::encode(self).ok().expect("encoding JSON failed")
    }

    /// Builds a new network from a JSON string.
    pub fn from_json(encoded: &str) -> Network {
        let network: Network = json::decode(encoded).ok().expect("decoding JSON failed");
        network
    }

    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push( utils::sigmoid(utils::modified_dotprod(&node, &results[layer_index])) )
            }
            results.push(layer_results);
        }
        results
    }

    // updates all weights in the network
    fn update_weights(&mut self, network_weight_updates: &Vec<Vec<Vec<f64>>>, prev_deltas: &mut Vec<Vec<Vec<f64>>>, rate: f64, momentum: f64) {
        for layer_index in 0..self.layers.len() {
            let mut layer = &mut self.layers[layer_index];
            let layer_weight_updates = &network_weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let mut node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                    let delta = (rate * weight_update) + (momentum * prev_delta);
                    node[weight_index] += delta;
                    prev_deltas[layer_index][node_index][weight_index] = delta;
                }
            }
        }

    }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }

        network_level
    }
}
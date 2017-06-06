extern crate time;

use super::Network;
use self::time::{ Duration, PreciseTime };

static DEFAULT_LEARNING_RATE: f64 = 0.3;
static DEFAULT_MOMENTUM: f64 = 0.;
static DEFAULT_MSE: f64 = 0.01;

/// Specifies when to stop training the network
#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    /// Stop training after a certain number of epochs
    Epochs(u32),
    /// Train until a certain error rate is achieved
    MSE(f64),
    /// Train for some fixed amount of time and then halt
    Timer(Duration),
}

/// Specifies which [learning mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to use when training the network
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    /// train the network Incrementally (updates weights after each example)
    Incremental
}

/// Used to specify options that dictate how a network will be trained
#[derive(Debug)]
pub struct Trainer<'a,'b> {
    examples: &'b [(Vec<f64>, Vec<f64>)],
    rate: f64,
    momentum: f64,
    log_interval: Option<u32>,
    halt_condition: self::HaltCondition,
    learning_mode: LearningMode,
    nn: &'a mut Network,
}

impl<'a,'b> Trainer<'a,'b>  {
    // TODO WRITE ABOUT THIS
    /// Takes in vector of examples and returns a `Trainer` struct that is used
    /// to specify options that dictate how the training should proceed.
    /// No actual training will occur until the `go()` method on the
    /// `Trainer` struct is called.
    pub fn new(target: &'a mut super::Network, examples: &'b [(Vec<f64>, Vec<f64>)]) -> Trainer<'a, 'b> {
        Trainer {
            examples: examples,
            rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            log_interval: None,
            halt_condition: HaltCondition::MSE(DEFAULT_MSE),
            learning_mode: LearningMode::Incremental,
            nn: target,
        }
    }

    /// Specifies the learning rate to be used when training (default is `0.3`)
    /// This is the step size that is used in the backpropagation algorithm.
    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a,'b> {
        if rate <= 0f64 {
            panic!("the learning rate must be a positive number");
        }

        self.rate = rate;
        self
    }

    /// Specifies the momentum to be used when training (default is `0.0`)
    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a,'b> {
        if momentum <= 0f64 {
            panic!("momentum must be positive");
        }

        self.momentum = momentum;
        self
    }

    /// Specifies how often (measured in batches) to log the current error rate (mean squared error) during training.
    /// `Some(x)` means log after every `x` batches and `None` means never log
    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a,'b> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => ()
        }

        self.log_interval = log_interval;
        self
    }

    /// Specifies when to stop training. `Epochs(x)` will stop the training after
    /// `x` epochs (one epoch is one loop through all of the training examples)
    /// while `MSE(e)` will stop the training when the error rate
    /// is at or below `e`. `Timer(d)` will halt after the [duration](https://doc.rust-lang.org/time/time/struct.Duration.html) `d` has
    /// elapsed.
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a,'b> {
        match halt_condition {
            HaltCondition::Epochs(epochs) if epochs < 1 => {
                panic!("must train for at least one epoch")
            }
            HaltCondition::MSE(mse) if mse <= 0f64 => {
                panic!("MSE must be greater than 0")
            }
            _ => ()
        }

        self.halt_condition = halt_condition;
        self
    }

    /// Specifies what [mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to train the network in.
    /// `Incremental` means update the weights in the network after every example.
    pub fn learning_mode(&mut self, learning_mode: LearningMode) -> &mut Trainer<'a,'b> {
        self.learning_mode = learning_mode;
        self
    }

    /// When `go` is called, the network will begin training based on the
    /// options specified. If `go` does not get called, the network will not
    /// get trained!
    pub fn go(&mut self) -> f64 {
        // check that input and output sizes are correct
        let input_layer_size = self.nn.num_inputs;
        let output_layer_size = self.nn.layers[self.nn.layers.len() - 1].len();
        for &(ref inputs, ref outputs) in self.examples.iter() {
            if inputs.len() as u32 != input_layer_size {
                panic!("input has a different length than the network's input layer");
            }
            if outputs.len() != output_layer_size {
                panic!("output has a different length than the network's output layer");
            }
        }

        let (examples, rate, momentum, log_interval, halt_condition) = (self.examples, self.rate, self.momentum, self.log_interval, self.halt_condition);
        self.train_incremental(examples, rate, momentum, log_interval, halt_condition)
    }

    fn train_incremental(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                         halt_condition: HaltCondition) -> f64 {

        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0u32;
        let mut training_error_rate = 0f64;
        let start_time = PreciseTime::now();

        loop {

            if epochs > 0 {
                // log error rate if necessary
                match log_interval {
                    Some(interval) if epochs % interval == 0 => {
                        println!("error rate: {}", training_error_rate);
                    },
                    _ => (),
                }

                // check if we've met the halt condition yet
                match halt_condition {
                    HaltCondition::Epochs(epochs_halt) => {
                        if epochs == epochs_halt { break }
                    },
                    HaltCondition::MSE(target_error) => {
                        if training_error_rate <= target_error { break }
                    },
                    HaltCondition::Timer(duration) => {
                        let now = PreciseTime::now();
                        if start_time.to(now) >= duration { break }
                    }
                }
            }

            training_error_rate = 0f64;

            for &(ref inputs, ref targets) in examples.iter() {
                let results = self.nn.do_run(&inputs);
                let weight_updates = self.calculate_weight_updates(&results, &targets);
                training_error_rate += super::utils::calculate_error(&results, &targets);
                self.nn.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
            }

            epochs += 1;
        }

        training_error_rate
    }

    // calculates all weight updates by backpropagation
    fn calculate_weight_updates(&self, results: &Vec<Vec<f64>>, targets: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let mut network_errors:Vec<Vec<f64>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.nn.layers;
        let network_results = &results[1..]; // skip the input layer
        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in super::utils::iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();


            for (node_index, (node, &result)) in super::utils::iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();
                //let mut node_error;
                let node_error;

                // calculate error for this node
                if layer_index == layers.len() - 1 {
                    node_error = result * (1f64 - result) * (targets[node_index] - result);
                } else {
                    let mut sum = 0f64;
                    let next_layer_errors = &network_errors[network_errors.len() - 1];
                    for (next_node, &next_node_error_data) in next_layer_nodes.unwrap().iter().zip((next_layer_errors).iter()) {
                        sum += next_node[node_index+1] * next_node_error_data; // +1 because the 0th weight is the threshold
                    }
                    node_error = result * (1f64 - result) * sum;
                }

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    //let mut prev_layer_result;
                    let prev_layer_result;
                    if weight_index == 0 {
                        prev_layer_result = 1f64; // threshold
                    } else {
                        prev_layer_result = prev_layer_results[weight_index-1];
                    }
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // updates were built by backpropagation so reverse them
        network_weight_updates.reverse();

        network_weight_updates
    }

    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new();
        for layer in self.nn.layers.iter() {
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
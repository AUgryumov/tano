//TODO WRITE DOC

use super::network::Network;

/// Trait to allows you to optimize your network
pub trait Optimizer<'a> {
    fn new(target: &'a mut Network, examples: &Vec<(Vec<f64>, Vec<f64>)>, epoch: &u32, learning_rate: &f64) -> Self;
    fn run(&mut self);
}
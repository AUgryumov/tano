use super::network::Network;

/// Trait to allows you to optimize your network
pub trait Optimizer<'a> {
    fn new(target: &'a mut Network, learning_rate: f64) -> Self;
    fn run(&mut self);
}

pub struct BackPropagationOptimizer<'a> {
    target: &'a mut Network,
    learning_rate: f64
}

impl<'a> Optimizer<'a> for BackPropagationOptimizer<'a>  {
    fn new(target: &'a mut Network, learning_rate: f64) -> Self {
        BackPropagationOptimizer::<'a> {
            target,
            learning_rate
        }
    }

    fn run(&mut self) {
        for layer in self.target.get_layers() {}
    }
}
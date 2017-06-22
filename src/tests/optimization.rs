use super::super::network::*;
use super::super::layers::{Layer, UsualLayer};
use super::super::optimizers::{Optimizer, OptimizationTypes};

#[test]
fn network() {
    let mut network = NetworkBuilder::new(2, 1)
        .layer(Box::new(UsualLayer::new(3)))
        .finalize();
    network.optimize(OptimizationTypes::FeedForward(vec![1., 1.], vec![0.], 0.1))
}
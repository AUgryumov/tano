extern crate tano;

use tano::network::NetworkBuilder;
use tano::layers::{Layer, UsualLayer};
use tano::optimizers::{Optimizer, OptimizationModes};

#[test]
fn xor() {
    let train_data = vec![
        (vec![0_f64, 0_f64], vec![0_f64]),
        (vec![1_f64, 0_f64], vec![1_f64]),
        (vec![0_f64, 1_f64], vec![1_f64]),
        (vec![1_f64, 1_f64], vec![0_f64]),
    ];

    let mut net = NetworkBuilder::new(2)
        .layer(Box::new(UsualLayer::new(3)))
        .layer(Box::new(UsualLayer::new(1)))
        .finalize();

    for (input, expected) in train_data {
        net.optimize(OptimizationModes::FeedForward {
            input: input,
            expected: expected,
            learning_rate: 0.1
        });
    }

    println!("{:?}", net.go(vec![1., 1.]));
}
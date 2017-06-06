extern crate tano;

use tano::*;
use tano::trainer::*;

fn log(epoch: u32, error_rate: f64) {
    if epoch % 1000 == 0 {
        println!("ERROR RATE: {}", error_rate);
    }
}

fn main() {
    // Let's take XOR examples
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];

    // Create a new neural network
    let mut net = Network::new(&[2, 3, 1]);

    // Train a network with out log function
    Trainer::new(&mut net, &examples).log(Some(log)).go();

    // You will see output like this:
    // ERROR RATE: 1.0251236782493696
    // ERROR RATE: 0.40451006715350024
    // ERROR RATE: 0.02400022295438454
    // ERROR RATE: 0.01078171039162239
}
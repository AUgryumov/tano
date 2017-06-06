extern crate tano;

use tano::*;
use tano::trainer::*;

fn main() {
    // Create examples of the xor function
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];

    // Create a new neural network
    let mut net1 = Network::new(&[2, 3, 1]);

    // Train a network
    Trainer::new(&mut net1, &examples).go();

    // Check a result
    for &(ref input, ref expected) in examples.iter() {
        let result = net1.run(&input)[0];
        let expected = expected[0];
        let status = if result.round() == expected { "OK" } else { "BAD" };
        println!("NETWORK OUT: {}; ROUNDED OUT: {}; EXPECTED OUT: {}; STATUS: {}", result, result.round(), expected, status);
    }
}
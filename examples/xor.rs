extern crate tano;

use tano::Network;

fn main() {
    let net = Network::new(&[1, 2, 3]);
    println!("{:?}", net.run(vec![3.]));
}
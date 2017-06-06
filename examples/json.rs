extern crate tano;

use tano::Network;

fn main() {
    // Let's create usual network
    let net = Network::new(&[1, 1]);

    // And encode it
    let json = net.to_json();
    println!("JSON: {}", net.to_json());

    // You can easily restore network data:
    let net_copy = Network::from_json(&json);
}

// You can write it to a file and share it between sessions.
// By this method you can train network on your computer and run it on a remote server.
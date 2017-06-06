# TANO
At this moment this is refactored RustNN fork with some new functions and examples.
It's not available in Cargo, but it will be available with first stable version.
## XOR sample
You can see full example [here](https://github.com/AUgryumov/tano/tree/master/examples/xor.rs).
More examples are available [here](https://github.com/AUgryumov/tano/tree/master/examples)
```rust
let examples = [
    (vec![0f64, 0f64], vec![0f64]),
    (vec![0f64, 1f64], vec![1f64]),
    (vec![1f64, 0f64], vec![1f64]),
    (vec![1f64, 1f64], vec![0f64]),
];

let mut net1 = Network::new(&[2, 3, 1]);

Trainer::new(&mut net1, &examples).go();
//Now you can result by net1.run(&input).
```
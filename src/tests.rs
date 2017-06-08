#[test]
fn network() {
    use super::layer::Layer;
    use super::layer::UsualLayer;
    use super::network::NetworkBuilder;

    let net1 = NetworkBuilder::new(3, 1).layer(Box::new(UsualLayer::new(2))).finalize();
    let net2 = NetworkBuilder::new(3, 1).layer(Box::new(UsualLayer::new(2))).finalize();

    let input = vec![0., 0., 0.];
    assert!(net1.run(&input) != net2.run(&input));
}

#[test]
fn usual_layer() {
    use super::layer::Layer;
    use super::layer::UsualLayer;
    use super::network::NetworkBuilder;

    let mut layer1 = UsualLayer::new(10);
    let mut layer2 = UsualLayer::new(100);

    assert!(layer1.get_weights() != layer2.get_weights());

    let net = NetworkBuilder::new(2, 1).layer(Box::new(layer1)).finalize();
    println!("{:?}", net.run(&vec![1_f64, 1_f64]));
    println!("{:?}", net);
}
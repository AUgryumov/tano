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

    let mut layer1 = UsualLayer::new(10);
    let mut layer2 = UsualLayer::new(100);

    assert!(layer1.get_neurons() != layer2.get_neurons());
}

#[test]
fn back_propagation_optimizer() {
    use super::optimizer::Optimizer;
    use super::optimizer::BackPropagationOptimizer;
    use super::network::NetworkBuilder;

    let network = &mut NetworkBuilder::new(1, 1).finalize();
    let mut optimizer = BackPropagationOptimizer::new(network, 0.1);
    optimizer.run();
}
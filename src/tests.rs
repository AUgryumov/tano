#[test]
fn usual_layer_creating() {
    use super::layers::UsualLayer;
    use super::layers::Layer;

    // Create a layers
    UsualLayer::new(1, 1);
    UsualLayer::new(1000, 1000);
}

#[test]
fn usual_layer_learning() {
    use super::layers::UsualLayer;
    use super::layers::Layer;

    // Create a neuron
    let mut layer = UsualLayer::new(1, 1);

    // Train a neuron
    for _ in 0..1000 {
        let actual = layer.run(&vec![1.])[0];
        let error = 1. - actual;
        assert!(!(actual.is_nan() || actual.is_infinite()));

        layer.train(vec![error], vec![1.], 0.01);
    }

    // Check result
    assert_eq!(layer.run(&vec![1.])[0].round(), 1_f64);
}
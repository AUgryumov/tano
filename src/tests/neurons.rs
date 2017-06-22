use super::super::neurons::{Neuron, UsualNeuron, RecurrentNeuron};
use super::super::utils::no_activation;

// TODO ADD TESTS WITH MORE WEIGHTS
#[test]
fn usual_neuron() {
    let mut neuron = UsualNeuron::new(1);
    neuron.get_mut_weights()[0] = 1.;
    assert_eq!(neuron.run(&vec![1.], no_activation), 1.);
    assert_eq!(neuron.run(&vec![0.], no_activation), 0.);

    neuron.get_mut_weights()[0] = 0.;
    assert_eq!(neuron.run(&vec![1.], no_activation), 0.);
    assert_eq!(neuron.run(&vec![0.], no_activation), 0.);

    neuron.get_mut_weights()[0] = 0.5;
    assert_eq!(neuron.run(&vec![1.], no_activation), 0.5);
    assert_eq!(neuron.run(&vec![0.], no_activation), 0.);
}

// TODO ADD TESTS WITH MORE WEIGHTS
#[test]
fn recurrent_neuron() {
    let mut neuron = RecurrentNeuron::new(1);
    {
        let weights = neuron.get_mut_weights();
        weights[0] = 1.;
        weights[1] = 1.;
    }
    for i in 1..10 {
        assert_eq!(neuron.run(&vec![1.], super::super::utils::no_activation), i as f64)
    }
}
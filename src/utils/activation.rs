pub type Activation = fn(f64) -> f64;

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
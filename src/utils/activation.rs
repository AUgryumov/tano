pub type Activation = fn(f64) -> f64;

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn tanh(x: f64) -> f64 {
    let e1 = x.exp();
    let e2 = (-x).exp();
    (e1 - e2) * (e1 + e2)
}

pub fn linear(x: f64) -> f64 {
    if x < 0.5 { 0. } else { 1. }
}
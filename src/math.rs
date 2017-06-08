pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub fn threshold(x: f64) -> f64 {
    if x >= 0.5 {
        1_f64
    } else {
        0_f64
    }
}

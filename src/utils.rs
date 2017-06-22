extern crate rand;

use utils::rand::{Rng, thread_rng};

pub type Activation = fn(f64) -> f64;

pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

pub(crate) fn gen_random_weight() -> f64 {
    thread_rng().gen()
}
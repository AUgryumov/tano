use std::iter::{Zip, Enumerate};
use std::slice;

pub fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + (-y).exp())
}
pub fn linear(y: f64) -> f64 { if y <= 0.5 { 0. } else { 1. } }

// calculates MSE of output layer
pub fn calculate_error(results: &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    let ref last_results = results[results.len()-1];
    let mut total:f64 = 0f64;
    for (&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f64)
}

pub fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

pub fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>  {
    s.iter().zip(t.iter()).enumerate()
}

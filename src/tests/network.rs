use super::super::network::*;
use super::super::layers::{Layer, UsualLayer};

// TODO WRITE
#[test]
fn network() {
    NetworkBuilder::new(2, 1)
        .layer(Box::new(UsualLayer::new(3)))
        .finalize()
        .go(vec![1., 1.]);
}
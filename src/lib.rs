pub mod nn;
pub mod util;

pub use nn::activation::Activation;
pub use nn::layer::Layer;
pub use nn::loss::LossFunction;
pub use nn::network::{Network, NetworkBuilder};
pub use util::vector::VecN;

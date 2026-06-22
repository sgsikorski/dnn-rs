use super::{activation::Activation, layer::Layer, loss::LossFunction};
use crate::util::vector::VecN;

pub struct Network {
    pub layers: Vec<Layer>,
    pub loss: LossFunction,
}

impl Network {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            loss: LossFunction::MSE,
        }
    }

    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new()
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn set_loss(&mut self, loss: LossFunction) {
        self.loss = loss;
    }

    pub fn forward(&mut self, input: &VecN) -> VecN {
        self.layers
            .iter_mut()
            .fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    pub fn predict(&mut self, input: &VecN) -> VecN {
        self.forward(input)
    }

    pub fn backward(&mut self, loss_grad: &VecN, lr: f64) {
        self.layers
            .iter_mut()
            .rev()
            .fold(loss_grad.clone(), |grad, layer| layer.backward(&grad, lr));
    }

    pub fn train_step(&mut self, input: &VecN, target: &VecN, lr: f64) -> f64 {
        let predicted = self.forward(input);
        let loss = self.loss.apply(&predicted, target);
        let grad = self.loss.grad(&predicted, target);
        self.backward(&grad, lr);
        loss
    }

    pub fn train(&mut self, samples: &[(VecN, VecN)], lr: f64) -> f64 {
        let total_loss: f64 = samples
            .iter()
            .map(|(input, target)| self.train_step(input, target, lr))
            .sum();
        total_loss / samples.len() as f64
    }

    pub fn train_epochs(&mut self, samples: &[(VecN, VecN)], lr: f64, epochs: usize) -> Vec<f64> {
        (0..epochs).map(|_| self.train(samples, lr)).collect()
    }

    pub fn evaluate(&mut self, samples: &[(VecN, VecN)]) -> f64 {
        let total_loss: f64 = samples
            .iter()
            .map(|(input, target)| {
                let predicted = self.forward(input);
                self.loss.apply(&predicted, target)
            })
            .sum();
        total_loss / samples.len() as f64
    }

    pub fn accuracy(&mut self, samples: &[(VecN, VecN)]) -> f64 {
        let correct = samples
            .iter()
            .filter(|(input, target)| {
                let predicted = self.forward(input);
                argmax(&predicted.data) == argmax(&target.data)
            })
            .count();
        correct as f64 / samples.len() as f64
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Network [")?;
        for layer in &self.layers {
            writeln!(f, "  {layer}")?;
        }
        write!(f, "] Loss: {}", self.loss)
    }
}

fn argmax(data: &[f64]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub struct NetworkBuilder {
    layer_configs: Vec<(usize, Activation)>,
    loss: LossFunction,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self {
            layer_configs: Vec::new(),
            loss: LossFunction::MSE,
        }
    }

    pub fn layer(mut self, size: usize, activation: Activation) -> Self {
        self.layer_configs.push((size, activation));
        self
    }

    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    pub fn build(self, input_size: usize) -> Network {
        let mut layers = Vec::new();
        let mut prev_size = input_size;
        for (size, activation) in self.layer_configs {
            layers.push(Layer::new(prev_size, size, activation));
            prev_size = size;
        }
        Network {
            layers,
            loss: self.loss,
        }
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

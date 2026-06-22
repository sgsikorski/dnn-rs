use super::activation::Activation;
use crate::util::{rng::Rng, vector::VecN};

pub struct Layer {
    pub weights: Vec<VecN>,
    pub biases: Vec<f64>,
    pub activation: Activation,
    last_input: Option<VecN>,
    last_pre_activation: Option<VecN>,
    last_output: Option<VecN>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = Rng::new();
        let scale = (6.0_f64 / (input_size + output_size) as f64).sqrt();
        let weights = (0..output_size)
            .map(|_| {
                VecN::new(
                    (0..input_size)
                        .map(|_| rng.gen_range(-scale, scale))
                        .collect(),
                )
            })
            .collect();

        Self {
            weights,
            biases: vec![0.0; output_size],
            activation,
            last_input: None,
            last_pre_activation: None,
            last_output: None,
        }
    }

    pub fn input_size(&self) -> usize {
        self.weights.first().map(|w| w.len()).unwrap_or(0)
    }

    pub fn output_size(&self) -> usize {
        self.weights.len()
    }

    pub fn forward(&mut self, input: &VecN) -> VecN {
        let pre_activations: Vec<f64> = self
            .weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w, b)| w.dot(input) + b)
            .collect();

        self.last_input = Some(input.clone());
        self.last_pre_activation = Some(VecN::new(pre_activations.clone()));

        let output = VecN::new(self.activation.apply_vec(&pre_activations));
        self.last_output = Some(output.clone());
        output
    }

    pub fn backward(&mut self, grad: &VecN, lr: f64) -> VecN {
        let last_input = self
            .last_input
            .as_ref()
            .expect("forward must be called before backward");
        let last_pre = self
            .last_pre_activation
            .as_ref()
            .expect("forward must be called before backward");

        let delta: Vec<f64> = if matches!(self.activation, Activation::Softmax) {
            let output = self
                .last_output
                .as_ref()
                .expect("forward must be called before backward");
            let dot_gs: f64 = grad
                .data
                .iter()
                .zip(output.data.iter())
                .map(|(g, s)| g * s)
                .sum();
            grad.data
                .iter()
                .zip(output.data.iter())
                .map(|(g, s)| s * (g - dot_gs))
                .collect()
        } else {
            grad.data
                .iter()
                .zip(last_pre.data.iter())
                .map(|(g, pre)| g * self.activation.derivative(*pre))
                .collect()
        };

        let mut input_grad = VecN::zeros(last_input.len());
        for (i, neuron_weights) in self.weights.iter().enumerate() {
            for (j, w) in neuron_weights.data.iter().enumerate() {
                input_grad.data[j] += delta[i] * w;
            }
        }

        for (i, neuron_weights) in self.weights.iter_mut().enumerate() {
            for (j, w) in neuron_weights.data.iter_mut().enumerate() {
                *w -= lr * delta[i] * last_input.data[j];
            }
            self.biases[i] -= lr * delta[i];
        }

        input_grad
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer({} -> {}, {})",
            self.input_size(),
            self.output_size(),
            self.activation
        )
    }
}

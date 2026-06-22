#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    ReLU,
    LeakyReLU(f64),
    ELU(f64),
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::ELU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Softmax => x,
            Activation::Linear => x,
        }
    }

    pub fn apply_vec(&self, xs: &[f64]) -> Vec<f64> {
        match self {
            Activation::Softmax => {
                let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = xs.iter().map(|&x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|e| e / sum).collect()
            }
            _ => xs.iter().map(|&x| self.apply(x)).collect(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Activation::ELU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    self.apply(x) + alpha
                }
            }
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Softmax => 1.0,
            Activation::Linear => 1.0,
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::ReLU => write!(f, "ReLU"),
            Activation::LeakyReLU(alpha) => write!(f, "LeakyReLU({alpha})"),
            Activation::ELU(alpha) => write!(f, "ELU({alpha})"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Softmax => write!(f, "Softmax"),
            Activation::Linear => write!(f, "Linear"),
        }
    }
}

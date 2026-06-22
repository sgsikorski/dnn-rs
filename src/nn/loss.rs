use crate::util::vector::VecN;

#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
}

impl LossFunction {
    pub fn apply(&self, predict: &VecN, actual: &VecN) -> f64 {
        match self {
            LossFunction::MSE => {
                let sum: f64 = (0..predict.len())
                    .map(|i| {
                        let e = actual[i] - predict[i];
                        e * e
                    })
                    .sum();
                sum / predict.len() as f64
            }
            LossFunction::CrossEntropy => {
                let eps = 1e-12;
                -(0..predict.len())
                    .map(|i| actual[i] * predict[i].max(eps).ln())
                    .sum::<f64>()
                    / predict.len() as f64
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                -(0..predict.len())
                    .map(|i| {
                        let p = predict[i].clamp(eps, 1.0 - eps);
                        actual[i] * p.ln() + (1.0 - actual[i]) * (1.0 - p).ln()
                    })
                    .sum::<f64>()
                    / predict.len() as f64
            }
        }
    }

    pub fn grad(&self, predict: &VecN, actual: &VecN) -> VecN {
        let n = predict.len() as f64;
        match self {
            LossFunction::MSE => VecN::new(
                (0..predict.len())
                    .map(|i| 2.0 * (predict[i] - actual[i]) / n)
                    .collect(),
            ),
            LossFunction::CrossEntropy => {
                let eps = 1e-12;
                VecN::new(
                    (0..predict.len())
                        .map(|i| -actual[i] / (predict[i].max(eps) * n))
                        .collect(),
                )
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = 1e-12;
                VecN::new(
                    (0..predict.len())
                        .map(|i| {
                            let p = predict[i].clamp(eps, 1.0 - eps);
                            (-(actual[i] / p) + (1.0 - actual[i]) / (1.0 - p)) / n
                        })
                        .collect(),
                )
            }
        }
    }
}

impl std::fmt::Display for LossFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossFunction::MSE => write!(f, "MSE"),
            LossFunction::CrossEntropy => write!(f, "CrossEntropy"),
            LossFunction::BinaryCrossEntropy => write!(f, "BinaryCrossEntropy"),
        }
    }
}

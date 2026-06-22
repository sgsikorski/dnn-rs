#[derive(Debug, Clone, PartialEq)]
pub struct VecN {
    pub data: Vec<f64>,
}

impl VecN {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    fn assert_same_len(&self, other: &VecN) {
        assert_eq!(self.len(), other.len(), "Vector lengths must match");
    }

    pub fn dot(&self, other: &VecN) -> f64 {
        self.assert_same_len(other);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn cross(&self, other: &VecN) -> VecN {
        assert_eq!(self.len(), 3, "Cross product requires 3D vectors");
        assert_eq!(other.len(), 3, "Cross product requires 3D vectors");
        VecN::new(vec![
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ])
    }

    pub fn magnitude(&self) -> f64 {
        self.data.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    }

    pub fn normalize(&self) -> VecN {
        let mag = self.magnitude();
        assert!(mag != 0.0, "Cannot normalize a zero vector");
        VecN::new(self.data.iter().map(|x| x / mag).collect())
    }
}

impl From<Vec<f64>> for VecN {
    fn from(data: Vec<f64>) -> Self {
        Self { data }
    }
}

impl From<VecN> for Vec<f64> {
    fn from(v: VecN) -> Vec<f64> {
        v.data
    }
}

use std::ops::{Add, Index, Mul, Neg, Sub};

impl Add for VecN {
    type Output = VecN;
    fn add(self, other: VecN) -> VecN {
        self.assert_same_len(&other);
        VecN::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl Sub for VecN {
    type Output = VecN;
    fn sub(self, other: VecN) -> VecN {
        self.assert_same_len(&other);
        VecN::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        )
    }
}

impl Mul<f64> for VecN {
    type Output = VecN;
    fn mul(self, scalar: f64) -> VecN {
        VecN::new(self.data.iter().map(|x| x * scalar).collect())
    }
}

impl Mul<VecN> for f64 {
    type Output = VecN;
    fn mul(self, v: VecN) -> VecN {
        VecN::new(v.data.iter().map(|x| x * self).collect())
    }
}

impl Mul<VecN> for VecN {
    type Output = VecN;
    fn mul(self, other: VecN) -> VecN {
        self.assert_same_len(&other);
        VecN::new(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
        )
    }
}

impl Neg for VecN {
    type Output = VecN;
    fn neg(self) -> VecN {
        VecN::new(self.data.iter().map(|x| -x).collect())
    }
}

impl Index<usize> for VecN {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

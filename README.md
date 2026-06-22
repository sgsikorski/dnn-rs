# rust-nn

A minimal deep neural network library in Rust. Supports dense layers, backpropagation, and common activations and loss functions.

## Building

```bash
cargo build
cargo test
```

## Importing into another project

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-nn = { path = "../rust-nn" }
```

Or via git:

```toml
[dependencies]
rust-nn = { git = "https://github.com/sgsikorski/rust-nn" }
```

## Usage

```rust
use rust_nn::{Network, Activation, LossFunction};

let mut net = Network::builder()
    .layer(128, Activation::ReLU)
    .layer(64, Activation::ReLU)
    .layer(10, Activation::Softmax)
    .loss(LossFunction::CrossEntropy)
    .build(784);  // input size

// Train
let losses = net.train_epochs(&samples, 0.001, 10);

// Evaluate
let accuracy = net.accuracy(&test_samples);  // assumes one-hot targets
let loss     = net.evaluate(&test_samples);

// Inference
let output = net.predict(&input);
```

Samples are `Vec<(VecN, VecN)>` where each pair is `(input, target)`.

```rust
use rust_nn::VecN;

let input  = VecN::new(vec![0.5, 0.2, 0.8]);
let target = VecN::new(vec![0.0, 1.0, 0.0]);
```

## Activations

`ReLU` · `LeakyReLU(f64)` · `ELU(f64)` · `Sigmoid` · `Tanh` · `Softmax` · `Linear`

## Loss functions

`MSE` · `CrossEntropy` · `BinaryCrossEntropy`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
cargo check          # type-check without linking (fast; use this first)
cargo build          # compile library + binary
cargo build --lib    # compile library only
cargo test           # run tests (none exist yet)
cargo clippy         # lint
cargo fmt            # format all source files
rustfmt --check src/ # check formatting without applying
```

## Architecture

This is a mixed crate: `src/lib.rs` exposes the neural network as a reusable library, and `src/main.rs` is a standalone binary that demonstrates training on MNIST.

**Important:** `main.rs` re-declares `mod nn`, `mod reader`, and `mod util` directly rather than importing via the library crate. This means the binary compiles the same source files as the library but as a separate crate root — library-level re-exports in `lib.rs` are not used by the binary.

### Module layout

```
src/
  lib.rs          — public re-exports: Network, NetworkBuilder, Layer, Activation, LossFunction, VecN
  main.rs         — MNIST training demo (re-declares modules directly)
  reader.rs       — MNIST CSV loader; binary-only, not in lib.rs
  nn/
    network.rs    — Network struct + NetworkBuilder
    layer.rs      — Layer struct (weights, biases, activation, cached forward state)
    activation.rs — Activation enum
    loss.rs       — LossFunction enum
    connection.rs — placeholder Connection struct (unused in dense layers)
    neuron.rs     — placeholder Neuron struct (empty)
  util/
    vector.rs     — VecN: n-dimensional f64 vector with ops
```

### Data flow

Forward pass: `Network::forward` folds input through each `Layer::forward`, which computes `weights·input + bias`, caches `last_input` and `last_pre_activation` (and `last_output` for Softmax), then applies the activation.

Backward pass: `Network::backward` folds the loss gradient through layers in reverse. Each `Layer::backward` computes deltas, accumulates the input gradient, and updates weights/biases in-place using vanilla SGD. Softmax backward uses the simplified Jacobian: `delta_j = s_j * (g_j - dot(g, s))`.

Training: `train_step` → `forward` → `loss.apply` + `loss.grad` → `backward`. The `last_input`/`last_pre_activation` fields on `Layer` are the only mutable state that bridges forward and backward; they must be populated before `backward` is called.

### Key design choices

- **`rand = "0.8"`** is the only external dependency, used in `Layer::new` for Xavier/Glorot weight initialization via `rand::thread_rng()`.
- **LossFunction and Activation are enums**, not trait objects. This avoids boxing and keeps dispatch monomorphic.
- **Xavier uniform initialization** (`±sqrt(6 / (fan_in + fan_out))`) is applied in `Layer::new`.
- **Softmax** requires `apply_vec` (operates on the full pre-activation slice) rather than the element-wise `apply`. `Layer::forward` always calls `apply_vec`, which delegates to `apply` element-wise for all other activations.
- **`accuracy`** uses argmax on both prediction and target vectors, so it assumes one-hot targets.
- **`VecN::data`** is public. Direct index access via `v[i]` is also available through the `Index<usize>` impl.

### Adding a new activation

Add a variant to the `Activation` enum in [activation.rs](src/nn/activation.rs), then handle it in `apply`, `apply_vec`, `derivative`, and `Display`. If the activation is not element-wise, handle it in `apply_vec` and return early from `derivative` (as Softmax does).

### Adding a new loss function

Add a variant to `LossFunction` in [loss.rs](src/nn/loss.rs) and implement `apply`, `grad`, and `Display` arms. The gradient must be with respect to the network output (not pre-activation).

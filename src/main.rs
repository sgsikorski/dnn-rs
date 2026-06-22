use nn::activation::Activation;
use nn::loss::LossFunction;
use nn::network::NetworkBuilder;

mod nn;
mod reader;
mod util;

fn main() {
    let dataset = reader::load_mnist("data/mnist_train.csv", "data/mnist_test.csv")
        .expect("Failed to load MNIST data");

    println!(
        "Loaded {} training samples and {} test samples",
        dataset.train.len(),
        dataset.test.len()
    );

    let mut net = NetworkBuilder::new()
        .layer(128, Activation::ReLU)
        .layer(64, Activation::ReLU)
        .layer(10, Activation::Softmax)
        .loss(LossFunction::CrossEntropy)
        .build(784);

    println!("{net}");

    let lr = 0.001;
    let losses = net.train_epochs(&dataset.train, lr, 5);
    for (epoch, loss) in losses.iter().enumerate() {
        println!("Epoch {}: avg loss = {:.6}", epoch + 1, loss);
    }

    let test_loss = net.evaluate(&dataset.test);
    let test_acc = net.accuracy(&dataset.test);
    println!(
        "Test loss: {test_loss:.6}  |  Test accuracy: {:.2}%",
        test_acc * 100.0
    );
}

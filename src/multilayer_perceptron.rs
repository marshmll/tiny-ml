use crate::layer::Layer;
use crate::neuron::ActivationType;
use crate::value::ValuePointer;

#[allow(unused)]
pub struct MultilayerPerceptron {
    layers: Vec<Layer>,
    learning_rate: f32,
}

#[allow(unused)]
impl MultilayerPerceptron {
    /// - `num_of_inputs`: The size of the input vector.
    /// - `layer_sizes`: A slice representing the number of neurons in each subsequent layer.
    /// - `learning_rate`: The step size for gradient descent.
    /// - `activation_type`: The neurons's activation type from the [ActivationType] enum
    pub fn new(
        num_of_inputs: usize,
        layer_sizes: &[usize],
        learning_rate: f32,
        activation_type: ActivationType,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_input_size = num_of_inputs;

        for &size in layer_sizes {
            // Create a layer: inputs = current_input_size, neurons = size
            layers.push(Layer::new(size, current_input_size, activation_type));

            // The output of this layer becomes the input for the next
            current_input_size = size;
        }

        MultilayerPerceptron {
            layers,
            learning_rate,
        }
    }

    pub fn forward(&mut self, inputs: &[ValuePointer]) -> Vec<ValuePointer> {
        let mut current_out = inputs.to_vec();

        for layer in &mut self.layers {
            current_out = layer.pass(&current_out);
        }

        current_out
    }

    pub fn parameters(&self) -> Vec<ValuePointer> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_gradients(&self) {
        for layer in &self.layers {
            layer.zero_gradients();
        }
    }

    /// Performs Stochastic Gradient Descent (SGD)
    /// Updates all parameters: p = p - learning_rate * grad
    pub fn update(&mut self) {
        let params = self.parameters();

        for p in params {
            let mut node = p.borrow_mut();
            let step = node.grad * self.learning_rate;
            node.data -= step;
        }
    }
}

#[cfg(test)]
mod mlp_tests {
    use crate::{
        multilayer_perceptron::MultilayerPerceptron, neuron::ActivationType, value::Value,
    };

    #[test]
    fn test_mlp_structure() {
        // Input: 3 features
        // Hidden Layer 1: 4 neurons
        // Output Layer: 1 neuron
        let mlp = MultilayerPerceptron::new(3, &[4, 1], 0.01, ActivationType::SIGM);

        assert_eq!(mlp.layers.len(), 2);

        // Layer 0: 4 neurons, each expecting 3 inputs
        assert_eq!(mlp.layers[0].parameters().len(), 4 * (3 + 1));

        // Layer 1: 1 neuron, expecting 4 inputs (from previous layer)
        assert_eq!(mlp.layers[1].parameters().len(), 1 * (4 + 1));
    }

    #[test]
    fn test_mlp_forward_pass() {
        let mut mlp = MultilayerPerceptron::new(2, &[2, 1], 0.1, ActivationType::SIGM);

        let inputs = vec![Value::new(1.0), Value::new(2.0)];
        let out = mlp.forward(&inputs);

        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_mlp_learning_step() {
        // Simple task: Teach the network to output 0.0 for input [1.0, 1.0]
        // This effectively tests backward pass and update logic
        let mut mlp = MultilayerPerceptron::new(2, &[2, 1], 0.1, ActivationType::SIGM);

        let inputs = vec![Value::new(1.0), Value::new(1.0)];

        // 1. Initial Forward Pass
        let out = mlp.forward(&inputs);
        let initial_prediction = out[0].borrow().data;

        // 2. Calculate Loss (MSE against target 0.0)
        // Loss = pred^2
        let loss = Value::pow(out[0].clone(), 2.0);

        // 3. Backward Pass
        mlp.zero_gradients();
        Value::backward(&loss);

        // 4. Update Weights
        mlp.update();

        // 5. Check if loss decreased
        let new_out = mlp.forward(&inputs);
        let new_prediction = new_out[0].borrow().data;
        let new_loss = new_prediction.powi(2);

        // The loss should have decreased after one step of gradient descent
        // (Unless we started exactly at 0, which is unlikely with random weights)
        if initial_prediction.abs() > 1e-5 {
            assert!(
                new_loss < initial_prediction.powi(2),
                "Loss did not decrease! Old: {}, New: {}",
                initial_prediction.powi(2),
                new_loss
            );
        }
    }
}

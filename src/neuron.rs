use rand::Rng;

use crate::value::{Value, ValuePointer};

#[allow(unused)]
#[derive(Debug)]
pub enum ActivationType {
    RELU,
    TANH,
    SIGM,
}

#[allow(unused)]
#[derive(Debug)]
pub struct Neuron {
    weights: Vec<ValuePointer>,
    bias: ValuePointer,
    activation_type: ActivationType,
}

#[allow(unused)]
impl Neuron {
    pub fn new(number_of_inputs: usize, activation_type: ActivationType) -> Self {
        let mut rng = rand::rng();

        let weights: Vec<ValuePointer> = (0..number_of_inputs)
            .map(|_| Value::new(rng.random_range(0.0_f32..=1.0_f32)))
            .collect();

        let bias = Value::new(0.0);

        Neuron {
            weights,
            bias,
            activation_type,
        }
    }

    pub fn parameters(&self) -> Vec<ValuePointer> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }

    pub fn zero_gradients(&self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }

    pub fn dot(&mut self, inputs: &[ValuePointer]) -> ValuePointer {
        assert_eq!(
            self.weights.len(),
            inputs.len(),
            "Weight count {} must match input count {}",
            self.weights.len(),
            inputs.len()
        );

        // Use an iterator to zip weights and inputs, then sum them up
        let mut act = self.bias.clone();

        for (w, x) in self.weights.iter().zip(inputs.iter()) {
            let mul = Value::mul(w.clone(), x.clone());
            act = Value::add(act, mul);
        }

        match self.activation_type {
            ActivationType::RELU => Value::relu(act),
            ActivationType::TANH => Value::tanh(act),
            ActivationType::SIGM => Value::sigm(act),
        }
    }
}

#[cfg(test)]
mod neuron_tests {
    use super::*;

    #[test]
    fn test_neuron_forward_relu() {
        // Create a neuron with 2 inputs
        let mut neuron = Neuron::new(2, ActivationType::RELU);

        // Manually set weights and bias for predictable results
        // Use a block to drop the borrow_mut immediately
        {
            neuron.weights[0].borrow_mut().data = 0.5;
            neuron.weights[1].borrow_mut().data = -1.0;
            neuron.bias.borrow_mut().data = 0.1;
        }

        // Inputs: x0 = 2.0, x1 = 1.0
        let x0 = Value::new(2.0);
        let x1 = Value::new(1.0);
        let inputs = vec![x0.clone(), x1.clone()];

        // Forward Pass:
        // out = ReLU((0.5 * 2.0) + (-1.0 * 1.0) + 0.1)
        // out = ReLU(1.0 - 1.0 + 0.1) = ReLU(0.1) = 0.1
        let out = neuron.dot(&inputs);

        assert!((out.borrow().data - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_backward() {
        let mut neuron = Neuron::new(2, ActivationType::RELU);

        // Set weights: w0=0.5, w1=0.2, bias=0.0
        {
            neuron.weights[0].borrow_mut().data = 0.5;
            neuron.weights[1].borrow_mut().data = 0.2;
            neuron.bias.borrow_mut().data = 0.0;
        }

        let x0 = Value::new(2.0);
        let x1 = Value::new(3.0);
        let inputs = vec![x0.clone(), x1.clone()];

        // Forward: (0.5*2.0) + (0.2*3.0) + 0.0 = 1.0 + 0.6 = 1.6
        // ReLU(1.6) = 1.6
        let out = neuron.dot(&inputs);
        Value::backward(&out);

        // Gradients:
        // d(out)/dw0 = x0 = 2.0
        // d(out)/dw1 = x1 = 3.0
        // d(out)/db  = 1.0
        assert_eq!(neuron.weights[0].borrow().grad, 2.0);
        assert_eq!(neuron.weights[1].borrow().grad, 3.0);
        assert_eq!(neuron.bias.borrow().grad, 1.0);

        // d(out)/dx0 = w0 = 0.5
        assert_eq!(x0.borrow().grad, 0.5);
    }

    #[test]
    fn test_neuron_zero_get_grad() {
        let mut neuron = Neuron::new(1, ActivationType::RELU);

        // Manually set weight to something positive so ReLU doesn't kill the gradient
        {
            neuron.weights[0].borrow_mut().data = 1.0;
            neuron.bias.borrow_mut().data = 0.0;
        }

        let x = vec![Value::new(1.0)];

        let out = neuron.dot(&x);
        Value::backward(&out);

        // Now this will reliably be 1.0, not 0.0
        assert!(neuron.weights[0].borrow().grad != 0.0);

        // Zero them
        neuron.zero_gradients();

        assert_eq!(neuron.weights[0].borrow().grad, 0.0);
        assert_eq!(neuron.bias.borrow().grad, 0.0);
    }

    #[test]
    fn test_neuron_tanh() {
        let mut neuron = Neuron::new(1, ActivationType::TANH);

        // Set weight=1.0, bias=0.0
        {
            neuron.weights[0].borrow_mut().data = 1.0;
            neuron.bias.borrow_mut().data = 0.0;
        }

        let x = vec![Value::new(0.5)];
        let out = neuron.dot(&x);

        // Forward: tanh(1.0 * 0.5 + 0.0) = tanh(0.5) ≈ 0.4621
        let expected_data = 0.5_f32.tanh();
        assert!((out.borrow().data - expected_data).abs() < 1e-6);

        Value::backward(&out);

        // Backward:
        // local_grad = 1 - tanh^2(0.5) ≈ 1 - (0.4621)^2 ≈ 0.7864
        // d(out)/dw = local_grad * x = 0.7864 * 0.5 = 0.3932
        let local_grad = 1.0 - expected_data.powi(2);
        let expected_grad = local_grad * 0.5;

        assert!((neuron.weights[0].borrow().grad - expected_grad).abs() < 1e-6);
    }

    #[test]
    fn test_neuron_sigmoid() {
        let mut neuron = Neuron::new(1, ActivationType::SIGM);

        // Set weight=2.0, bias=0.0
        {
            neuron.weights[0].borrow_mut().data = 2.0;
            neuron.bias.borrow_mut().data = 0.0;
        }

        // Input x=0.0 makes the dot product 0.0
        let x = vec![Value::new(0.0)];
        let out = neuron.dot(&x);

        // Forward: sigm(0) = 0.5
        assert_eq!(out.borrow().data, 0.5);

        Value::backward(&out);

        // Backward:
        // local_grad = sigm(0) * (1 - sigm(0)) = 0.5 * 0.5 = 0.25
        // d(out)/dbias = local_grad * 1.0 = 0.25
        assert_eq!(neuron.bias.borrow().grad, 0.25);
    }
}

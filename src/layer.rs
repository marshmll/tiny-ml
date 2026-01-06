use crate::{
    neuron::{ActivationType, Neuron},
    value::ValuePointer,
};

#[allow(unused)]
#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub num_of_inputs_per_neuron: usize,
}

#[allow(unused)]
impl Layer {
    pub fn new(
        num_of_neurons: usize,
        num_of_inputs_per_neuron: usize,
        activation_type: ActivationType,
    ) -> Self {
        let neurons = (0..num_of_neurons)
            .map(|_| Neuron::new(num_of_inputs_per_neuron, activation_type))
            .collect();

        Layer {
            neurons,
            num_of_inputs_per_neuron,
        }
    }

    pub fn pass(&mut self, inputs: &[ValuePointer]) -> Vec<ValuePointer> {
        self.neurons.iter_mut().map(|n| n.dot(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<ValuePointer> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn zero_gradients(&self) {
        for n in &self.neurons {
            n.zero_gradients();
        }
    }
}

#[cfg(test)]
mod layer_tests {
    use crate::{layer::Layer, neuron::ActivationType, value::Value};

    #[test]
    fn test_layer_forward_dimensions() {
        let num_neurons = 3;
        let num_inputs = 2;
        let mut layer = Layer::new(num_neurons, num_inputs, ActivationType::RELU);

        let inputs = vec![Value::new(1.0), Value::new(0.5)];
        let outputs = layer.pass(&inputs);

        // A layer with 3 neurons should return 3 output values
        assert_eq!(outputs.len(), 3);
    }

    #[test]
    fn test_layer_parameters_count() {
        let num_neurons = 4;
        let num_inputs = 3;
        let layer = Layer::new(num_neurons, num_inputs, ActivationType::SIGM);

        let params = layer.parameters();

        // Each neuron has (num_inputs) weights + 1 bias
        // 4 neurons * (3 weights + 1 bias) = 16 parameters
        assert_eq!(params.len(), num_neurons * (num_inputs + 1));
    }

    #[test]
    fn test_layer_backward_flow() {
        // Create a simple layer: 2 neurons, 1 input each
        let mut layer = Layer::new(2, 1, ActivationType::RELU);

        // Manually set weights for predictable math
        // Neuron 0: w=2.0, b=0.0
        // Neuron 1: w=-1.0, b=0.0
        {
            layer.neurons[0].weights[0].borrow_mut().data = 2.0;
            layer.neurons[1].weights[0].borrow_mut().data = -1.0;
            for n in &layer.neurons {
                n.bias.borrow_mut().data = 0.0;
            }
        }

        let input = vec![Value::new(10.0)];
        let outputs = layer.pass(&input);

        // Neuron 0: 2.0 * 10.0 = 20.0 (ReLU stays 20.0)
        // Neuron 1: -1.0 * 10.0 = -10.0 (ReLU becomes 0.0)
        assert_eq!(outputs[0].borrow().data, 20.0);
        assert_eq!(outputs[1].borrow().data, 0.0);

        // Backward from both outputs
        for out in &outputs {
            crate::value::Value::backward(out);
        }

        // Neuron 0 weight grad: d(20)/dw = input = 10.0
        assert_eq!(layer.neurons[0].weights[0].borrow().grad, 10.0);
        // Neuron 1 weight grad: d(0)/dw = 0.0 (ReLU was dead)
        assert_eq!(layer.neurons[1].weights[0].borrow().grad, 0.0);
    }
}

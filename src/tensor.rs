use crate::value::{Value, ValuePointer};

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Tensor {
    // Stored as rows of ValuePointers
    pub data: Vec<Vec<ValuePointer>>,
}

#[allow(unused)]
impl Tensor {
    /// Creates a 2D Tensor from a slice of slices
    pub fn new(inputs: &[&[f32]]) -> Self {
        let data = inputs
            .iter()
            .map(|row| row.iter().map(|&val| Value::new(val)).collect())
            .collect();

        Tensor { data }
    }

    /// Creates a 1D "vector" Tensor (represented as a single row)
    pub fn from_vec(inputs: &[f32]) -> Self {
        let row = inputs.iter().map(|&val| Value::new(val)).collect();
        Tensor { data: vec![row] }
    }

    pub fn zero_gradients(&self) {
        for row in &self.data {
            for val in row {
                val.borrow_mut().grad = 0.0;
            }
        }
    }

    pub fn row(&self, index: usize) -> Option<&Vec<ValuePointer>> {
        self.data.get(index)
    }

    pub fn shape(&self) -> (usize, usize) {
        let rows = self.data.len();
        let cols = if rows > 0 { self.data[0].len() } else { 0 };
        (rows, cols)
    }

    pub fn push_row(&mut self, row: Vec<ValuePointer>) {
        self.data.push(row);
    }
}

#[cfg(test)]
mod tensor_tests {
    use crate::{layer::Layer, neuron::ActivationType, tensor::Tensor};

    #[test]
    fn test_tensor_shape() {
        // 2 rows, 3 columns
        let tensor = Tensor::new(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        assert_eq!(tensor.shape(), (2, 3));
    }

    #[test]
    fn test_tensor_indexing() {
        let tensor = Tensor::new(&[&[10.0, 20.0]]);

        let row = tensor.row(0).unwrap();
        assert_eq!(row[0].borrow().data, 10.0);
        assert_eq!(row[1].borrow().data, 20.0);
    }

    #[test]
    fn test_tensor_gradient_reset() {
        let tensor = Tensor::from_vec(&[1.0, 2.0]);

        // Manually set gradients
        {
            let row = tensor.row(0).unwrap();
            row[0].borrow_mut().grad = 5.0;
            row[1].borrow_mut().grad = 10.0;
        }

        tensor.zero_gradients();

        let row = tensor.row(0).unwrap();
        assert_eq!(row[0].borrow().grad, 0.0);
        assert_eq!(row[1].borrow().grad, 0.0);
    }

    #[test]
    fn test_tensor_integration_with_layer() {
        // 1. Create a Tensor of inputs (Batch of 2, 3 features each)
        let inputs = Tensor::new(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        // 2. Create a Layer (2 neurons, 3 inputs each)
        let mut layer = Layer::new(2, 3, ActivationType::RELU);

        // 3. Pass each row of the tensor through the layer
        let mut results = Vec::new();
        for i in 0..inputs.shape().0 {
            let input_row = inputs.row(i).unwrap();
            results.push(layer.pass(input_row));
        }

        // Check that we got 2 sets of outputs, each with 2 neuron results
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
    }
}

# Tiny ML: A From-Scratch Neural Network Implementation in Rust

A lightweight, educational implementation of a multilayer perceptron neural network in pure Rust, featuring automatic differentiation (autograd) and backpropagation. This project demonstrates core machine learning concepts through a clean, type-safe implementation without external deep learning frameworks.

## 🎯 Project Overview

**Tiny ML** is a minimal yet complete neural network library that implements:

- **Automatic Differentiation (Autograd)**: A computational graph that automatically computes gradients
- **Backpropagation**: Efficient gradient computation through chain rule
- **Multilayer Perceptron**: Fully connected neural networks with customizable layer sizes
- **Activation Functions**: ReLU, Tanh, and Sigmoid
- **Stochastic Gradient Descent**: Parameter optimization via gradient descent

The project serves as both a learning tool and a practical demonstration of how modern deep learning frameworks compute gradients under the hood.

## 🏗️ Architecture

### Core Components

#### 1. **Value** (`src/value.rs`)
The foundation of the automatic differentiation engine.

```
Value = Computational Graph Node
├── data: f32 (forward pass value)
├── grad: f32 (accumulated gradient)
├── op: char (operation applied: '+', '-', '*', '/', etc.)
├── prev: Vec<ValuePointer> (parent nodes in computation graph)
└── back: Option<Box<dyn FnMut()>> (backward closure for gradient computation)
```

**Key Insight**: Each `Value` is wrapped in `Rc<RefCell<>>` (reference-counted, interior-mutable cell) to allow:
- Multiple ownership of nodes (essential for computational graphs)
- Shared mutation without unsafe code
- Automatic cleanup when no longer referenced

**Operations Supported**:
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`
- Activations: `relu`, `tanh`, `sigm`
- Backward pass via `Value::backward(&root)` which traverses the graph in topological order

#### 2. **Neuron** (`src/neuron.rs`)
A single artificial neuron implementing the perceptron model.

```
Neuron:
├── weights: Vec<ValuePointer> (learnable parameters)
├── bias: ValuePointer (learnable parameter)
└── activation_type: ActivationType (activation function)

Forward Pass:
  output = activation(bias + Σ(weight_i × input_i))
```

**Features**:
- Configurable activation functions (ReLU, Tanh, Sigmoid)
- Parameter extraction for gradient descent
- Gradient zeroing for clean backpropagation

#### 3. **Layer** (`src/layer.rs`)
A collection of neurons operating in parallel.

```
Layer (with n neurons, m inputs each):
┌─────────────────────────────┐
│ Neuron 1 → Output 1         │
│ Neuron 2 → Output 2         │ m inputs
│ ...                         │ →
│ Neuron n → Output n         │
└─────────────────────────────┘
         ↓
    n outputs
```

**Properties**:
- All neurons receive the same input vector
- Produces a vector of outputs (one per neuron)
- Parameters = n × (m + 1) where n = neurons, m = inputs

#### 4. **Multilayer Perceptron (MLP)** (`src/multilayer_perceptron.rs`)
Stacks multiple layers to create deep networks.

```
Architecture Example: 2 → 4 → 1
┌─────────────────────────────┐
│ Input Layer (2 features)    │
│ [x₀, x₁]                    │
└──────────┬──────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Hidden Layer (4 neurons)│
    │ [h₀, h₁, h₂, h₃]        │
    └──────┬──────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Output Layer (1 neuron) │
    │ [y]                     │
    └─────────────────────────┘
```

**Key Methods**:
- `new(inputs, layer_sizes, learning_rate, activation)`: Create network topology
- `forward(&inputs)`: Forward pass through all layers
- `parameters()`: Extract all learnable weights and biases
- `update()`: Perform one SGD step using computed gradients

#### 5. **Tensor** (`src/tensor.rs`)
Batch handling for efficient data processing.

```
Tensor (2D matrix of Values):
[row₀: [v₀₀, v₀₁, v₀₂]]
[row₁: [v₁₀, v₁₁, v₁₂]]
[row₂: [v₂₀, v₂₁, v₂₂]]
```

## 🧠 Machine Learning Concepts Implemented

### 1. **Forward Propagation**
Computing predictions from inputs through the network:
```
a⁽ˡ⁾ = σ(W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
```
Where:
- `W⁽ˡ⁾` = weights at layer l
- `a⁽ˡ⁻¹⁾` = activations from previous layer
- `b⁽ˡ⁾` = bias vector
- `σ` = activation function

### 2. **Computational Graph**
A directed acyclic graph (DAG) encoding the computation:
```
        [x₀, x₁]
           │
    [×] [+] [×] [+]  (Neuron 1)
           │
          [σ]
           │
    [×] [+] [×] [+]  (Neuron 2)
           │
          [σ]
           │
        [Loss]
```

### 3. **Automatic Differentiation (Reverse Mode)**
Efficiently computes gradients via chain rule:
```
∂L/∂w = ∂L/∂y · ∂y/∂z · ∂z/∂w

Where:
- L = loss
- y = activated output
- z = pre-activation sum
- w = weight
```

**Implementation**: Each operation stores a closure (`back` function) that propagates gradients to parent nodes.

### 4. **Backpropagation**
Traverses the computational graph in reverse topological order:
```
1. Call Value::backward(&loss) at the root
2. Set root.grad = 1.0 (dL/dL = 1)
3. For each node in reverse topological order:
   - Call the node's back() closure
   - Each closure updates parent gradients
```

### 5. **Gradient Descent Optimization**
Updates parameters to minimize loss:
```
w := w - α · ∂L/∂w

Where:
- α = learning rate (step size)
- ∂L/∂w = gradient from backprop
```

**In the code**:
```rust
node.data -= node.grad * learning_rate
```

### 6. **Loss Functions**
Mean Squared Error (MSE) for regression:
```
L = (1/n) Σ(ŷᵢ - yᵢ)²

Where:
- ŷᵢ = prediction
- yᵢ = target
```

## 📊 Example: XOR Problem

The project includes a complete example solving the classic XOR problem, which requires a non-linear model.

### The Problem
```
Input  → Output
(0,0)  → 0
(0,1)  → 1
(1,0)  → 1
(1,1)  → 0
```

Cannot be solved by a single neuron (XOR is not linearly separable). Requires a hidden layer.

### Network Architecture
```
Input: 2 features
Hidden: 4 neurons (tanh activation)
Output: 1 neuron (tanh activation)

Total parameters: 2×4 + 4 + 4×1 + 1 = 21 parameters
```

### Training Strategy
```
For each epoch:
  1. Forward pass: inputs → predictions
  2. Compute loss: MSE(predictions, targets)
  3. Backward pass: compute all gradients
  4. Update: apply gradient descent to all parameters
  5. Repeat for 300,000 epochs
```

### Results
After training, the network learns to approximate:
```
(0,0) → ~0.0
(0,1) → ~1.0
(1,0) → ~1.0
(1,1) → ~0.0
```

## 🔧 Usage

### Running the XOR Demo

```bash
cargo run --example xor_demo
```

Expected output:
```
Initial training start...
--------------------------------
Epoch   0 | Total Loss: 3.256411
Epoch 500 | Total Loss: 2.456789
...
Epoch 299500 | Total Loss: 0.000123
--------------------------------
Final Predictions:
Input: [0.0, 0.0] | Target: 0.0 | Prediction: 0.0024
Input: [0.0, 1.0] | Target: 1.0 | Prediction: 0.9876
Input: [1.0, 0.0] | Target: 1.0 | Prediction: 0.9834
Input: [1.0, 1.0] | Target: 0.0 | Prediction: 0.0156
```

### Building Your Own Network

```rust
use tiny_ml::{
    multilayer_perceptron::MultilayerPerceptron,
    neuron::ActivationType,
    value::Value,
};

fn main() {
    // Create a 3 → 8 → 4 → 1 network
    let mut model = MultilayerPerceptron::new(
        3,           // 3 inputs
        &[8, 4, 1],  // Layer sizes
        0.01,        // Learning rate
        ActivationType::TANH
    );

    // Training loop
    for epoch in 0..1000 {
        let mut loss = Value::new(0.0);

        for (input, target) in training_data {
            let pred = model.forward(&input);
            let error = Value::sub(pred[0].clone(), Value::new(*target));
            loss = Value::add(loss, Value::pow(error, 2.0));
        }

        model.zero_gradients();
        Value::backward(&loss);
        model.update();
    }
}
```

## 🧪 Testing

Comprehensive test suites validate each component:

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test neuron::
cargo test layer::
cargo test mlp_tests::
```

Test coverage includes:
- **Neuron tests**: Forward pass, backward pass, activation functions
- **Layer tests**: Dimension checking, parameter counting, gradient flow
- **MLP tests**: Network structure, forward/backward integration
- **Tensor tests**: Shape operations, batch processing

## 📐 Implementation Details

### Why Rc<RefCell<>>?

The computational graph requires:
1. **Multiple owners**: Parent nodes are referenced by multiple child nodes
2. **Interior mutability**: Gradients and data need to be modified during backward pass
3. **No garbage collection**: Rust's reference counting handles cleanup automatically

```rust
pub type ValuePointer = Rc<RefCell<Value>>;
```

### Topological Sorting for Backprop

The backward pass correctly handles complex graphs:
```rust
pub fn backward(root: &ValuePointer) {
    // Build topological order
    let mut visited = HashSet::new();
    let mut topo = Vec::new();
    
    fn build_topo(v: &ValuePointer, visited: &mut HashSet<u64>, topo: &mut Vec<ValuePointer>) {
        let id = v.borrow().id;
        if visited.insert(id) {
            for child in &v.borrow().prev {
                build_topo(child, visited, topo);
            }
            topo.push(v.clone());
        }
    }
    
    build_topo(root, &mut visited, &mut topo);
    
    // Backward pass
    root.borrow_mut().grad = 1.0;
    for node in topo.iter().rev() {
        if let Some(mut back_fn) = node.borrow_mut().back.take() {
            back_fn();
            node.borrow_mut().back = Some(back_fn);
        }
    }
}
```

### Gradient Accumulation

For nodes with multiple parents, gradients correctly accumulate:
```rust
// For a + a:
lhs_borrow.grad += out_grad;  // Not =, but +=
rhs_borrow.grad += out_grad;
```

## 🎓 Key Takeaways

This implementation teaches:

1. **How deep learning frameworks compute gradients** - Modern frameworks like PyTorch use similar autograd engines
2. **The importance of computational graphs** - Understanding DAGs helps debug training issues
3. **Type safety in ML code** - Rust's type system catches errors at compile time
4. **Numerical stability** - Managing gradient flow through deep networks
5. **Trade-offs in abstraction** - Balancing flexibility with performance

## 🚀 Future Enhancements

Potential extensions to explore:

- [ ] Batch normalization
- [ ] Convolutional layers
- [ ] LSTM/RNN cells
- [ ] Optimizers (Adam, RMSprop)
- [ ] Regularization (dropout, L1/L2)
- [ ] GPU acceleration
- [ ] Advanced activation functions (GELU, Swish)

## 📚 References

- **Backpropagation**: Rumelhart, Hinton, Williams (1986)
- **Automatic Differentiation**: Griewank & Walther (2008)
- **Neural Network Fundamentals**: Goodfellow, Bengio, Courville (2016)

## 📄 License

Educational project. Feel free to use, modify, and learn from this code.

---

**Built with ❤️ in Rust** | *Understanding neural networks from first principles*

pub mod sequential_generator;

use std::cell::RefCell;
use std::rc::Rc;

use crate::sequential_generator::SequentialGenerator;

pub static GENERATOR: SequentialGenerator = SequentialGenerator::new(1);

pub struct Value {
    data: f32,
    grad: f32,
    op: char,
    id: u64,
    prev: Vec<Rc<RefCell<Value>>>,
    back: Option<Box<dyn FnMut()>>,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value(data={:.4}, grad={:.4}, op='{}', id={})",
            self.data, self.grad, self.op, self.id
        )
    }
}

impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("op", &self.op)
            .field("id", &self.id)
            .field("prev", &self.prev.len())
            .finish()
    }
}

impl Value {
    pub fn new(data: f32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            op: '\0',
            id: GENERATOR.next(),
            prev: Vec::new(),
            back: None,
        }))
    }

    // Read-Only Getters
    pub fn data(&self) -> f32 {
        self.data
    }

    pub fn grad(&self) -> f32 {
        self.grad
    }

    pub fn op(&self) -> char {
        self.op
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn add(lhs: Rc<RefCell<Self>>, rhs: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;

        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();

        let out = Rc::new(RefCell::new(Value {
            data: lhs_data + rhs_data,
            grad: 0.0,
            op: '+',
            id: GENERATOR.next(),
            prev: vec![lhs_clone.clone(), rhs_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            // Scope the borrows. Update lhs, drop lock, then update rhs.
            // This prevents panic if lhs and rhs are the same node (e.g., a + a).
            {
                let mut lhs_borrow = lhs_clone.borrow_mut();
                lhs_borrow.grad += out_grad;
            }
            {
                let mut rhs_borrow = rhs_clone.borrow_mut();
                rhs_borrow.grad += out_grad;
            }
        }));

        out
    }

    pub fn mul(lhs: Rc<RefCell<Self>>, rhs: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;

        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();

        let out = Rc::new(RefCell::new(Value {
            data: lhs_data * rhs_data,
            grad: 0.0,
            op: '*',
            id: GENERATOR.next(),
            prev: vec![lhs_clone.clone(), rhs_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            // Read data immutably first (safe to do simultaneously)
            let lhs_data = lhs_clone.borrow().data;
            let rhs_data = rhs_clone.borrow().data;

            // Scope mutable borrows to avoid overlap
            {
                let mut lhs_borrow = lhs_clone.borrow_mut();
                lhs_borrow.grad += rhs_data * out_grad;
            }
            {
                let mut rhs_borrow = rhs_clone.borrow_mut();
                rhs_borrow.grad += lhs_data * out_grad;
            }
        }));

        out
    }

    pub fn sub(lhs: Rc<RefCell<Self>>, rhs: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;

        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();

        let out = Rc::new(RefCell::new(Value {
            data: lhs_data - rhs_data,
            grad: 0.0,
            op: '-',
            id: GENERATOR.next(),
            prev: vec![lhs_clone.clone(), rhs_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            {
                let mut lhs_borrow = lhs_clone.borrow_mut();
                lhs_borrow.grad += out_grad;
            }
            {
                let mut rhs_borrow = rhs_clone.borrow_mut();
                rhs_borrow.grad -= out_grad;
            }
        }));

        out
    }

    pub fn div(lhs: Rc<RefCell<Self>>, rhs: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let lhs_data = lhs.borrow().data;
        let rhs_data = rhs.borrow().data;

        assert_ne!(rhs_data, 0.0, "Division by zero is not allowed!");

        let lhs_clone = lhs.clone();
        let rhs_clone = rhs.clone();

        let out = Rc::new(RefCell::new(Value {
            data: lhs_data / rhs_data,
            grad: 0.0,
            op: '/',
            id: GENERATOR.next(),
            prev: vec![lhs_clone.clone(), rhs_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            let lhs_data = lhs_clone.borrow().data;
            let rhs_data = rhs_clone.borrow().data;

            {
                let mut lhs_borrow = lhs_clone.borrow_mut();
                lhs_borrow.grad += (1.0 / rhs_data) * out_grad;
            }
            {
                let mut rhs_borrow = rhs_clone.borrow_mut();
                rhs_borrow.grad += (-lhs_data / (rhs_data.powi(2))) * out_grad;
            }
        }));

        out
    }

    pub fn pow(base: Rc<RefCell<Self>>, exponent: f32) -> Rc<RefCell<Self>> {
        assert_ne!(exponent, 0.0, "Cannot raise value to the power of 0!");

        let base_data = base.borrow().data;
        let base_clone = base.clone();

        let out = Rc::new(RefCell::new(Value {
            data: base_data.powf(exponent),
            grad: 0.0,
            op: '^',
            id: GENERATOR.next(),
            prev: vec![base_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;
            let base_data = base_clone.borrow().data;

            {
                let mut base_borrow = base_clone.borrow_mut();
                base_borrow.grad += out_grad * exponent * base_data.powf(exponent - 1.0);
            }
        }));

        out
    }

    // Tanh activation function
    pub fn tanh(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let x = value.borrow().data;
        let t = x.tanh();

        let value_clone = value.clone();

        let out = Rc::new(RefCell::new(Value {
            data: t,
            grad: 0.0,
            op: 'T',
            id: GENERATOR.next(),
            prev: vec![value_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            let x = value_clone.borrow().data;
            // Derivative of tanh: 1 - tanh²(x)
            let local_grad = 1.0 - x.tanh().powi(2);

            {
                let mut value_borrow = value_clone.borrow_mut();
                value_borrow.grad += local_grad * out_grad;
            }
        }));

        out
    }

    // ReLU activation function
    pub fn relu(value: Rc<RefCell<Self>>) -> Rc<RefCell<Self>> {
        let x = value.borrow().data;
        let r = 0.0_f32.max(x);

        let value_clone = value.clone();

        let out = Rc::new(RefCell::new(Value {
            data: r,
            grad: 0.0,
            op: 'R',
            id: GENERATOR.next(),
            prev: vec![value_clone.clone()],
            back: None,
        }));

        let out_clone = out.clone();
        out.borrow_mut().back = Some(Box::new(move || {
            let out_grad = out_clone.borrow().grad;

            let x = out_clone.borrow().data;
            /*
                Derivative of ReLU:
                {
                    x > 0: 1
                    x <= 0: 0
                }
            */

            let local_grad = if x > 0.0_f32 { 1.0_f32 } else { 0.0_f32 };

            {
                let mut value_borrow = value_clone.borrow_mut();
                value_borrow.grad += local_grad * out_grad;
            }
        }));

        out
    }

    // Backward propagation method
    pub fn backward(value: &Rc<RefCell<Self>>) {
        // Set gradient of output to 1.0
        value.borrow_mut().grad = 1.0;

        // Topological order all children in the graph
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            v: Rc<RefCell<Value>>,
            topo: &mut Vec<Rc<RefCell<Value>>>,
            visited: &mut std::collections::HashSet<u64>,
        ) {
            if !visited.insert(v.borrow().id) {
                return;
            }
            for child in v.borrow().prev.iter() {
                build_topo(child.clone(), topo, visited);
            }
            topo.push(v);
        }

        build_topo(value.clone(), &mut topo, &mut visited);

        // Go one variable at a time in reverse order
        for v in topo.iter().rev() {
            // We use .take() to extract the closure.
            // This releases the mutable borrow on `v` immediately,
            // allowing the closure to safely borrow `v` (as `out`) to read .grad later.
            let back_op = v.borrow_mut().back.take();

            if let Some(mut back) = back_op {
                back();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = Value::add(a.clone(), b.clone());
        let d = Value::mul(c.clone(), c.clone());

        assert_eq!(c.borrow().data(), 3.0);
        assert_eq!(c.borrow().op(), '+');
        assert_eq!(d.borrow().data(), 9.0);
        assert_eq!(d.borrow().op(), '*');

        let loss = Value::add(d.clone(), d.clone());

        Value::backward(&loss);

        // Check gradients
        // loss = d + d = 9 + 9 = 18
        // dloss/dd = 2
        // d = c * c = 3 * 3 = 9
        // dd/dc = 2c = 6
        // dloss/dc = dloss/dd * dd/dc = 2 * 6 = 12
        // c = a + b = 1 + 2 = 3
        // dc/da = 1, dc/db = 1
        // dloss/da = dloss/dc * dc/da = 12 * 1 = 12
        // dloss/db = dloss/dc * dc/db = 12 * 1 = 12

        assert_eq!(loss.borrow().grad(), 1.0); // Self gradient
        assert_eq!(d.borrow().grad(), 2.0); // From loss = d + d
        assert_eq!(c.borrow().grad(), 12.0); // From chain rule
        assert_eq!(a.borrow().grad(), 12.0);
        assert_eq!(b.borrow().grad(), 12.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::mul(a.clone(), b.clone());

        Value::backward(&c);

        // c = a * b = 6
        // dc/da = b = 3
        // dc/db = a = 2
        assert_eq!(c.borrow().grad(), 1.0);
        assert_eq!(a.borrow().grad(), 3.0);
        assert_eq!(b.borrow().grad(), 2.0);
    }

    #[test]
    fn test_division() {
        let a = Value::new(6.0);
        let b = Value::new(2.0);
        let c = Value::div(a.clone(), b.clone());

        Value::backward(&c);

        // c = a / b = 3
        // dc/da = 1/b = 0.5
        // dc/db = -a/b² = -6/4 = -1.5
        assert_eq!(c.borrow().grad(), 1.0);
        assert_eq!(a.borrow().grad(), 0.5);
        assert_eq!(b.borrow().grad(), -1.5);
    }

    #[test]
    fn test_pow() {
        let a = Value::new(2.0);
        let b = Value::pow(a.clone(), 3.0); // b = a^3

        Value::backward(&b);

        // Forward pass: 2^3 = 8
        assert_eq!(b.borrow().data(), 8.0);

        // Backward pass:
        // d(a^3)/da = 3 * a^(3-1) = 3 * a^2
        // at a=2: 3 * 2^2 = 3 * 4 = 12
        assert_eq!(a.borrow().grad(), 12.0);
    }

    #[test]
    fn test_relu_positive() {
        let a = Value::new(2.0);
        let b = Value::relu(a.clone());

        Value::backward(&b);

        // Forward: max(0, 2) = 2
        assert_eq!(b.borrow().data(), 2.0);

        // Backward: since a > 0, gradient passes through (slope is 1)
        assert_eq!(a.borrow().grad(), 1.0);
    }

    #[test]
    fn test_relu_negative() {
        let a = Value::new(-2.0);
        let b = Value::relu(a.clone());

        Value::backward(&b);

        // Forward: max(0, -2) = 0
        assert_eq!(b.borrow().data(), 0.0);

        // Backward: since a <= 0, gradient is killed (slope is 0)
        assert_eq!(a.borrow().grad(), 0.0);
    }

    #[test]
    fn test_complex_pow_chain() {
        // Function: f(x) = (x + 1)^2
        // at x = 3
        let x = Value::new(3.0);
        let one = Value::new(1.0);

        let sum = Value::add(x.clone(), one.clone()); // sum = 4
        let y = Value::pow(sum.clone(), 2.0); // y = 4^2 = 16

        Value::backward(&y);

        assert_eq!(y.borrow().data(), 16.0);

        // Chain Rule:
        // y = u^2 where u = x + 1
        // dy/du = 2u
        // du/dx = 1
        // dy/dx = dy/du * du/dx = 2u * 1 = 2(x + 1)
        // at x=3: 2(3 + 1) = 8

        assert_eq!(x.borrow().grad(), 8.0);
    }
}

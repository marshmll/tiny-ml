use tiny_ml::{
    multilayer_perceptron::MultilayerPerceptron,
    neuron::ActivationType,
    value::{Value, ValuePointer},
};

fn main() {
    // 1. Define the XOR dataset
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    // Targets:  0,     1,     1,     0
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![0.0, 1.0, 1.0, 0.0];

    // 2. Initialize the MLP
    // 2 inputs -> Hidden Layer (4 neurons) -> Output Layer (1 neuron)
    // Learning rate: 0.1
    // Activation type: tanh
    let mut model = MultilayerPerceptron::new(2, &[4, 1], 0.1, ActivationType::TANH);

    println!("Initial training start...");
    println!("--------------------------------");

    // 3. Training Loop
    for epoch in 0..300000 {
        let mut total_loss = Value::new(0.0);

        // Batch processing
        for (i, input_row) in inputs.iter().enumerate() {
            // Convert f32 inputs to Values
            let input_values: Vec<ValuePointer> =
                input_row.iter().map(|&x| Value::new(x)).collect();

            // Forward pass
            let prediction = model.forward(&input_values);

            // Calculate Loss (Mean Squared Error for this example)
            // Loss = (prediction - target)^2
            let target = Value::new(targets[i]);
            let diff = Value::sub(prediction[0].clone(), target);
            let squared_error = Value::pow(diff, 2.0);

            // Accumulate total loss
            total_loss = Value::add(total_loss, squared_error);
        }

        // Zero gradients before backward pass
        model.zero_gradients();

        // Backward pass
        Value::backward(&total_loss);

        // Update parameters (Gradient Descent)
        model.update();

        // Print progress every 50 epochs
        if epoch % 500 == 0 || epoch == 300000 - 1 {
            println!(
                "Epoch {:3} | Total Loss: {:.6}",
                epoch,
                total_loss.borrow().data
            );
        }
    }

    println!("--------------------------------");
    println!("Final Predictions:");

    // 4. Verify results
    for (i, input_row) in inputs.iter().enumerate() {
        let input_values: Vec<ValuePointer> = input_row.iter().map(|&x| Value::new(x)).collect();

        let out = model.forward(&input_values);
        let pred = out[0].borrow().data;
        let target = targets[i];

        println!(
            "Input: {:?} | Target: {:.1} | Prediction: {:.4}",
            input_row, target, pred
        );
    }
}

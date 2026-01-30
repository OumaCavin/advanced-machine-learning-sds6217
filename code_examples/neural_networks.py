"""
Neural Networks Implementation

This module provides implementations of neural network components including
forward propagation, activation functions, and basic architectures.

Author: Cavin Otieno Ouma
Registration: SDS6/46982/2024
Course: SDS 6217 Advanced Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt


class ActivationFunction:
    """
    Collection of common activation functions used in neural networks.
    """
    
    @staticmethod
    def relu(z):
        """
        Rectified Linear Unit (ReLU).
        Most commonly used activation function in modern deep learning.
        
        f(z) = max(0, z)
        """
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        """
        Derivative of ReLU.
        
        f'(z) = 1 if z > 0, 0 otherwise
        """
        return (z > 0).astype(float)
    
    @staticmethod
    def sigmoid(z):
        """
        Sigmoid activation function.
        Maps values to (0, 1) range, useful for binary classification.
        
        f(z) = 1 / (1 + exp(-z))
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(z):
        """
        Derivative of sigmoid.
        
        f'(z) = sigmoid(z) * (1 - sigmoid(z))
        """
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z):
        """
        Hyperbolic Tangent activation.
        Maps values to (-1, 1) range, zero-centered.
        
        f(z) = tanh(z)
        """
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        """
        Derivative of tanh.
        
        f'(z) = 1 - tanh²(z)
        """
        return 1 - np.tanh(z) ** 2
    
    @staticmethod
    def softmax(z):
        """
        Softmax activation for multi-class classification.
        Converts logits to probability distribution.
        
        f(z_i) = exp(z_i) / Σ exp(z_j)
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NeuralNetworkLayer:
    """
    Base class for a neural network layer.
    """
    
    def __init__(self, n_input, n_output, activation='relu'):
        """
        Initialize a fully connected layer.
        
        Args:
            n_input: Number of input features
            n_output: Number of output neurons
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.n_input = n_input
        self.n_output = n_output
        
        # Xavier/Glorot initialization for better gradient flow
        limit = np.sqrt(6 / (n_input + n_output))
        self.weights = np.random.uniform(-limit, limit, (n_input, n_output))
        self.bias = np.zeros((1, n_output))
        
        # Store activation function
        self.activation_name = activation
        if activation == 'relu':
            self.activation = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        else:
            self.activation = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)
    
    def forward(self, X):
        """
        Perform forward propagation.
        
        Computes: Z = X @ W + b, then A = activation(Z)
        
        This demonstrates the "nonlinear transformation" during forward propagation.
        """
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, d_a, learning_rate):
        """
        Perform backward propagation.
        
        Args:
            d_a: Gradient of loss with respect to output
            learning_rate: Learning rate for weight updates
        """
        # Gradient of activation
        d_z = d_a * self.activation_derivative(self.z)
        
        # Gradients for weights and bias
        d_w = np.dot(self.a.T, d_z) / len(d_a)
        d_b = np.sum(d_z, axis=0, keepdims=True) / len(d_z)
        
        # Gradient for previous layer
        d_a_prev = np.dot(d_z, self.weights.T)
        
        # Update parameters (gradient descent)
        self.weights -= learning_rate * d_w
        self.bias -= learning_rate * d_b
        
        return d_a_prev


class NeuralNetwork:
    """
    Multi-layer Neural Network implementation.
    
    Demonstrates the three-stage learning process:
    1. Input Computation (forward propagation)
    2. Iterative Refinement (backpropagation + weight updates)
    3. Output Generation (final forward pass for predictions)
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize neural network with specified architecture.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for gradient descent
        """
        self.layers = []
        self.learning_rate = learning_rate
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            activation = 'relu' if i < len(layer_sizes) - 2 else 'sigmoid'
            layer = NeuralNetworkLayer(
                layer_sizes[i], layer_sizes[i + 1], activation
            )
            self.layers.append(layer)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        This is "Input Computation" and "Output Generation" combined.
        Data flows through each layer, undergoing:
        - Linear transformation: Z = X @ W + b
        - Nonlinear transformation: A = activation(Z)
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_true, y_pred):
        """
        Backpropagation to compute gradients.
        
        This is part of "Iterative Refinement" in the learning process.
        """
        # Start with gradient of loss (MSE derivative)
        d_a = 2 * (y_pred - y_true) / len(y_true)
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            d_a = layer.backward(d_a, self.learning_rate)
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network.
        
        This implements the complete "Iterative Refinement" process:
        1. Forward pass (compute predictions)
        2. Compute loss
        3. Backward pass (compute gradients)
        4. Update weights and biases
        5. Repeat
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training iterations
            verbose: Print progress
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(y, y_pred)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Generate predictions (output generation).
        """
        return self.forward(X)


class CNN:
    """
    Convolutional Neural Network for image processing.
    
    CNNs are specialized neural networks designed for image processing,
    as mentioned in the quiz: "CNN is a specialized artificial neural
    network designed for image processing."
    """
    
    @staticmethod
    def convolve2d(image, kernel, stride=1, padding=0):
        """
        Apply 2D convolution to an image.
        
        This is the core operation that makes CNNs effective for images:
        - Preserves spatial relationships
        - Detects local features (edges, textures)
        - Uses shared weights (parameter efficient)
        
        Args:
            image: Input image (H, W) or (C, H, W)
            kernel: Convolution kernel/filter
            stride: Step size for convolution
            padding: Zero padding around image
            
        Returns:
            Convolved feature map
        """
        # Add padding
        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding)))
        
        # Get dimensions
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]  # Add channel dimension
        
        in_channels, h, w = image.shape
        k_h, k_w = kernel.shape
        
        # Output dimensions
        out_h = (h - k_h) // stride + 1
        out_w = (w - k_w) // stride + 1
        
        # Apply convolution
        output = np.zeros((out_h, out_w))
        
        for i in range(0, out_h * stride, stride):
            for j in range(0, out_w * stride, stride):
                for c in range(in_channels):
                    patch = image[c, i:i+k_h, j:j+k_w]
                    output[i//stride, j//stride] += np.sum(patch * kernel)
        
        return output[0] if output.shape[0] == 1 else output
    
    @staticmethod
    def max_pool(feature_map, pool_size=2, stride=2):
        """
        Apply max pooling operation.
        
        Benefits:
        - Reduces spatial dimensions
        - Provides translation invariance
        - Reduces computation
        """
        h, w = feature_map.shape
        out_h = h // stride
        out_w = w // stride
        
        pooled = np.zeros((out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                patch = feature_map[i*stride:i*stride+pool_size,
                                   j*stride:j*stride+pool_size]
                pooled[i, j] = np.max(patch)
        
        return pooled


class LSTM:
    """
    Long Short-Term Memory network.
    
    As mentioned in the quiz: "LSTM is a variation of the RNN that
    introduces a memory mechanism to overcome the vanishing gradient problem."
    
    LSTMs solve the vanishing gradient problem through:
    1. Cell state (memory) that allows gradients to flow unchanged
    2. Gate mechanisms (forget, input, output) that control information flow
    3. Additive updates instead of multiplicative
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM parameters.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units (memory cells)
        """
        self.hidden_size = hidden_size
        
        # Xavier initialization for better gradient flow
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Forget gate parameters
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bf = np.zeros((1, hidden_size))
        
        # Input gate parameters
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bi = np.zeros((1, hidden_size))
        
        # Cell candidate parameters
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bc = np.zeros((1, hidden_size))
        
        # Output gate parameters
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.bo = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        """Numerical stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-np.clip(x, -500, 500))),
                       np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500))))
    
    def forward(self, x_sequence):
        """
        Forward pass through LSTM for a sequence.
        
        The key innovation is the cell state (memory) that maintains
        information over long sequences without vanishing gradients.
        
        Args:
            x_sequence: Input sequence (seq_len, input_size)
            
        Returns:
            Hidden states and cell state
        """
        seq_len = len(x_sequence)
        
        # Initialize
        h = np.zeros((seq_len + 1, self.hidden_size))
        c = np.zeros((seq_len + 1, self.hidden_size))
        
        # Store for backward pass
        self.x_sequence = x_sequence
        self.h = h
        self.c = c
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            h_t_minus_1 = h[t]
            
            # Concatenate input and previous hidden state
            combined = np.concatenate([x_t, h_t_minus_1])
            
            # Forget gate: What to discard from cell state
            f_t = self.sigmoid(combined @ self.Wf + self.bf)
            
            # Input gate: What new information to store
            i_t = self.sigmoid(combined @ self.Wi + self.bi)
            
            # Cell candidate: New candidate values
            c_tilde_t = np.tanh(combined @ self.Wc + self.bc)
            
            # Update cell state: f_t * c[t-1] + i_t * c_tilde_t
            c[t + 1] = f_t * c[t] + i_t * c_tilde_t
            
            # Output gate: What to output
            o_t = self.sigmoid(combined @ self.Wo + self.bo)
            
            # Hidden state: o_t * tanh(c[t+1])
            h[t + 1] = o_t * np.tanh(c[t + 1])
        
        return h[1:], c[1:]  # Return without initial zero state


def demonstrate_activation_functions():
    """
    Visualize different activation functions and their derivatives.
    """
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'ReLU': (ActivationFunction.relu, ActivationFunction.relu_derivative),
        'Sigmoid': (ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative),
        'Tanh': (ActivationFunction.tanh, ActivationFunction.tanh_derivative)
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for idx, (name, (func, deriv)) in enumerate(activations.items()):
        # Activation function
        ax = axes[idx, 0]
        ax.plot(x, func(x), 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'{name} Activation', fontsize=12)
        ax.set_xlabel('z')
        ax.set_ylabel(f'f(z)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 2)
        
        # Derivative
        ax = axes[idx, 1]
        ax.plot(x, deriv(x), 'r-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'{name} Derivative', fontsize=12)
        ax.set_xlabel('z')
        ax.set_ylabel(f"f'(z)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/activation_functions.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Activation functions visualization saved.")


def demonstrate_forward_propagation():
    """
    Demonstrate the forward propagation process in neural networks.
    
    This shows how data flows through layers, undergoing:
    1. Linear transformation (Z = X @ W + b)
    2. Nonlinear transformation (activation function)
    """
    print("\n" + "=" * 60)
    print("Forward Propagation Demonstration")
    print("=" * 60)
    
    # Simple example
    np.random.seed(42)
    
    # Input: 3 samples, 4 features
    X = np.random.randn(3, 4)
    
    # Create a simple network: 4 -> 3 -> 2 -> 1
    network = NeuralNetwork([4, 3, 2, 1], learning_rate=0.1)
    
    print("\nInput (X):")
    print(X)
    print(f"Shape: {X.shape}")
    
    # Manual forward pass demonstration
    print("\n" + "-" * 60)
    print("Manual Forward Pass:")
    print("-" * 60)
    
    # Layer 1: 4 -> 3
    z1 = X @ network.layers[0].weights + network.layers[0].bias
    a1 = ActivationFunction.relu(z1)
    print(f"\nLayer 1 (4 -> 3):")
    print(f"  Z = X @ W + b: {z1.shape}")
    print(f"  A = ReLU(Z): {a1.shape}")
    print(f"  (This is the 'activation' and 'nonlinear transformation')")
    
    # Layer 2: 3 -> 2
    z2 = a1 @ network.layers[1].weights + network.layers[1].bias
    a2 = ActivationFunction.relu(z2)
    print(f"\nLayer 2 (3 -> 2):")
    print(f"  Z = A1 @ W + b: {z2.shape}")
    print(f"  A = ReLU(Z): {a2.shape}")
    
    # Layer 3: 2 -> 1 (output)
    z3 = a2 @ network.layers[2].weights + network.layers[2].bias
    a3 = ActivationFunction.sigmoid(z3)
    print(f"\nLayer 3 (2 -> 1, Output):")
    print(f"  Z = A2 @ W + b: {z3.shape}")
    print(f"  A = Sigmoid(Z): {a3.shape}")
    
    # Automated forward pass
    print("\n" + "-" * 60)
    print("Automated Forward Pass (using network.forward()):")
    print("-" * 60)
    prediction = network.forward(X)
    print(f"\nPredictions: {prediction.flatten()}")
    
    print("\n" + "=" * 60)
    print("Key Takeaways from Forward Propagation:")
    print("=" * 60)
    print("1. Input flows through layers sequentially")
    print("2. Each layer applies: Z = X @ W + b (linear)")
    print("3. Then applies activation function (nonlinear)")
    print("4. This is 'Input Computation' -> 'Output Generation'")
    print("=" * 60)


def demonstrate_cnn_concepts():
    """
    Demonstrate CNN concepts for image processing.
    """
    print("\n" + "=" * 60)
    print("CNN (Convolutional Neural Network) Demonstration")
    print("=" * 60)
    
    print("\nAs stated in the quiz:")
    print("'CNN is a specialized artificial neural network")
    print(" designed for image processing.'")
    
    print("\nKey CNN Components:")
    print("-" * 60)
    print("1. Convolutional Layers:")
    print("   - Apply filters/kernels to detect features")
    print("   - Preserve spatial relationships")
    print("   - Shared weights reduce parameters")
    
    print("\n2. Pooling Layers:")
    print("   - Downsample feature maps")
    print("   - Provide translation invariance")
    print("   - Reduce computational load")
    
    print("\n3. Hierarchical Features:")
    print("   - Layer 1: Edges, simple textures")
    print("   - Layer 2: Shapes, patterns")
    print("   - Layer 3: Complex objects")
    
    # Create sample image (8x8 grayscale)
    image = np.zeros((8, 8))
    image[2:6, 2:6] = 1  # Square in center
    image[3:5, 3:5] = 0  # Hole in center
    
    # Edge detection kernel
    edge_kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
    
    # Apply convolution
    convolved = CNN.convolve2d(image, edge_kernel)
    
    print("\n" + "-" * 60)
    print("Convolution Example:")
    print("-" * 60)
    print("\nInput Image (8x8):")
    print(image)
    print("\nEdge Detection Kernel:")
    print(edge_kernel)
    print("\nConvolved Feature Map:")
    print(np.round(convolved, 2))
    
    print("\n" + "=" * 60)


def demonstrate_lstm_memory():
    """
    Demonstrate LSTM's memory mechanism.
    """
    print("\n" + "=" * 60)
    print("LSTM (Long Short-Term Memory) Demonstration")
    print("=" * 60)
    
    print("\nAs stated in the quiz:")
    print("'LSTM is a variation of the RNN that introduces a")
    print(" memory mechanism to overcome the vanishing gradient problem.'")
    
    print("\nLSTM Key Components:")
    print("-" * 60)
    print("1. Cell State (Memory):")
    print("   - Long-term information storage")
    print("   - Gradients can flow unchanged (solving vanishing gradient)")
    
    print("\n2. Gates:")
    print("   - Forget Gate: What to discard from memory")
    print("   - Input Gate: What new information to store")
    print("   - Output Gate: What to output based on memory")
    
    print("\n3. Vanishing Gradient Solution:")
    print("   - Additive updates: C_new = f*C_old + i*C_tilde")
    print("   - Gradients add instead of multiply")
    print("   - Information preserved over long sequences")
    
    # Simple sequence
    np.random.seed(42)
    sequence_length = 10
    input_size = 5
    hidden_size = 3
    
    # Create LSTM
    lstm = LSTM(input_size, hidden_size)
    
    # Generate random sequence
    x_seq = np.random.randn(sequence_length, input_size)
    
    # Forward pass
    h, c = lstm.forward(x_seq)
    
    print(f"\nLSTM Example:")
    print(f"  Input sequence: {sequence_length} timesteps, {input_size} features")
    print(f"  Hidden state size: {hidden_size}")
    print(f"  Output hidden states shape: {h.shape}")
    print(f"  Cell state shape: {c.shape}")
    
    print("\n" + "-" * 60)
    print("Compare with Vanilla RNN:")
    print("-" * 60)
    print("Vanilla RNN: Gradients diminish over long sequences")
    print("LSTM: Cell state preserves information over time")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Networks Demonstration")
    print("Course: SDS 6217 Advanced Machine Learning")
    print("Student: Cavin Otieno Ouma")
    print("=" * 60)
    
    # Activation functions
    print("\n1. Visualizing Activation Functions...")
    demonstrate_activation_functions()
    
    # Forward propagation
    print("\n2. Demonstrating Forward Propagation...")
    demonstrate_forward_propagation()
    
    # CNN concepts
    print("\n3. Demonstrating CNN Concepts...")
    demonstrate_cnn_concepts()
    
    # LSTM memory
    print("\n4. Demonstrating LSTM Memory Mechanism...")
    demonstrate_lstm_memory()
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)

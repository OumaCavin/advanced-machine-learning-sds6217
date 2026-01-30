"""
Linear Regression Implementation

This module demonstrates the implementation of linear regression using gradient descent,
covering concepts from the Advanced Machine Learning course (SDS 6217).

Author: Cavin Otieno Ouma
Registration: SDS6/46982/2024
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression model using gradient descent optimization.
    
    This implementation demonstrates:
    - Parameter initialization (weights and bias)
    - Gradient computation
    - Iterative parameter updates
    - Learning rate effects
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent updates
            n_iterations: Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Train the linear regression model.
        
        The model learns the optimal weights and bias by minimizing
        the Mean Squared Error (MSE) loss function using gradient descent.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent iterations
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def demonstrate_learning_rate_effects():
    """
    Demonstrate the effect of different learning rates on convergence.
    
    This shows why the learning rate should be "problem dependent"
    and how too high or too low rates affect training.
    """
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx // 2, idx % 2]
        
        # Train model with different learning rates
        model = LinearRegression(learning_rate=lr, n_iterations=100)
        model.fit(X, y)
        
        # Plot loss curve
        ax.plot(model.loss_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'Learning Rate = {lr}')
        ax.grid(True, alpha=0.3)
        
        # Add annotation for convergence behavior
        if lr == 0.001:
            behavior = "Too Slow"
        elif lr == 0.01:
            behavior = "Good"
        elif lr == 0.1:
            behavior = "Fast but unstable"
        else:
            behavior = "Diverges"
        ax.annotate(behavior, xy=(50, model.loss_history[50]), 
                   xytext=(60, model.loss_history[50] + 10),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/learning_rate_effects.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nLearning rate demonstration plot saved.")


def demonstrate_gradient_descent():
    """
    Visualize the gradient descent process on a 2D loss surface.
    """
    # Define a simple quadratic loss function
    def loss_function(w, b):
        return w**2 + b**2
    
    def grad_w(w, b):
        return 2 * w
    
    def grad_b(w, b):
        return 2 * b
    
    # Starting point
    w, b = 3, 3
    learning_rate = 0.1
    path = [(w, b)]
    
    # Run gradient descent
    for _ in range(20):
        w_new = w - learning_rate * grad_w(w, b)
        b_new = b - learning_rate * grad_b(w, b)
        w, b = w_new, b_new
        path.append((w, b))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contour
    w_range = np.linspace(-4, 4, 100)
    b_range = np.linspace(-4, 4, 100)
    W, B = np.meshgrid(w_range, b_range)
    Z = W**2 + B**2
    
    contour = ax.contour(W, B, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient descent path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'r-o', linewidth=2, markersize=8, 
            label='Gradient Descent Path')
    ax.scatter(path[0, 0], path[0, 1], c='green', s=200, marker='*', 
               label='Start', zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=200, marker='*', 
               label='End (Minimum)', zorder=5)
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_title('Gradient Descent on Quadratic Loss Surface')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/advanced-machine-learning-sds6217/code_examples/gradient_descent_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Gradient descent visualization saved.")


if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression Demonstration")
    print("Course: SDS 6217 Advanced Machine Learning")
    print("Student: Cavin Otieno Ouma")
    print("=" * 60)
    
    # Demonstrate gradient descent visualization
    print("\n1. Creating gradient descent visualization...")
    demonstrate_gradient_descent()
    
    # Demonstrate learning rate effects
    print("\n2. Creating learning rate effects demonstration...")
    demonstrate_learning_rate_effects()
    
    # Simple example
    print("\n3. Training a simple linear regression model...")
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_weights = np.array([3.5, -2.0])
    true_bias = 1.5
    y = np.dot(X, true_weights) + true_bias + np.random.randn(100) * 0.5
    
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\nLearned weights: {model.weights}")
    print(f"True weights: {true_weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"True bias: {true_bias}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")
    print(f"R-squared: {model.score(X, y):.4f}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)

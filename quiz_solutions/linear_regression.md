# Linear Regression Quiz Solutions

This document provides detailed explanations for the linear regression questions from the Advanced Machine Learning quiz.

---

## Question 1: What parts of the linear regression equation are updated during training?

**Answer: The bias and weights**

### Explanation

In linear regression, the model is represented by the equation:

**y = wx + b**

Where:
- **w** (weights) represents the coefficients multiplying each feature
- **b** (bias) represents the intercept term
- **x** represents the feature values (input data)
- **y** represents the prediction (output)

During training, only the parameters **w** and **b** are updated. The feature values (x) are fixed inputs from the dataset and never change. The prediction (y) is the model's output based on current parameter values.

The training process uses optimization algorithms like gradient descent to iteratively adjust w and b to minimize the loss function, typically Mean Squared Error (MSE):

**MSE = (1/n) * Σ(y_pred - y_actual)²**

---

## Question 2: What is the role of gradient descent in linear regression?

**Answer: Gradient descent is an iterative process that finds the best weights and bias that minimize the loss**

### Explanation

Gradient descent is an optimization algorithm that plays a crucial role in training linear regression models:

1. **Initialization**: Start with random initial values for weights and bias

2. **Gradient Computation**: Calculate the gradient of the loss function with respect to each parameter

   - ∂MSE/∂w = (2/n) * Σ(x * (w*x + b - y))
   - ∂MSE/∂b = (2/n) * Σ(w*x + b - y)

3. **Parameter Update**: Adjust parameters in the opposite direction of the gradient

   - w_new = w_old - learning_rate * ∂MSE/∂w
   - b_new = b_old - learning_rate * ∂MSE/∂b

4. **Iteration**: Repeat steps 2-3 until convergence

Gradient descent does NOT:
- Remove outliers from the dataset
- Determine which loss function to use (this is a design choice)
- Work automatically without proper learning rate selection

---

## Question 3: What is the ideal learning rate?

**Answer: Problem dependent**

### Explanation

The ideal learning rate is highly problem-dependent and depends on several factors:

1. **Dataset Characteristics**: Scale of features, noise level, size of dataset
2. **Model Complexity**: Number of parameters, architecture
3. **Loss Landscape**: Curvature, presence of saddle points, local minima
4. **Optimization Algorithm**: Different algorithms (SGD, Adam, RMSprop) have different recommended learning rates

**Common approaches to finding the right learning rate:**

- **Learning Rate Scheduling**: Start with a higher rate and decrease over time
- **Grid Search**: Try multiple values and select the best performing one
- **Cyclical Learning Rates**: Vary learning rate within a range during training
- **Adaptive Optimizers**: Use algorithms like Adam that adjust learning rates automatically

Typical learning rates range from 0.0001 to 1.0, with 0.01 being a common default for many problems.

---

## Question 4: Which of the following statements is true?

**Answer: Doubling the learning rate can slow down training**

### Explanation

This statement is true because:

1. **Too High Learning Rate**: Can cause overshooting - the algorithm jumps past the optimal solution and may even diverge
2. **Oscillation**: The loss may oscillate instead of decreasing monotonically
3. **Divergence**: In extreme cases, the loss can increase rather than decrease

Consider this example:
- Optimal step size: 0.1
- Current position: 5.0
- Gradient: -2.0
- Correct update: 5.0 + 0.1*2.0 = 5.2 (moving toward minimum)

With doubled learning rate (0.2):
- Update: 5.0 + 0.2*2.0 = 5.4 (overshooting the minimum)

The other options:
- "Larger batches are unsuitable for data with many outliers" - Not necessarily true; larger batches can actually provide more stable gradient estimates

---

## Question 5: What is the best batch size when using mini-batch SGD?

**Answer: It depends**

### Explanation

The optimal batch size depends on multiple factors:

1. **Memory Constraints**: Larger batches require more GPU/CPU memory
2. **Dataset Size**: Very small datasets may benefit from smaller batches
3. **Convergence Behavior**: Smaller batches provide more frequent updates but with higher variance
4. **Generalization**: Smaller batches often lead to better generalization due to noise

**Trade-offs:**

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (1-32) | Better generalization, more frequent updates | Noisy gradients, slower on GPU |
| Medium (32-256) | Balance of speed and stability | Requires hyperparameter tuning |
| Large (512+) | Stable gradients, fast computation | May converge to sharp minima |

**Common Practice**: Start with batch sizes of 32, 64, or 128 and adjust based on performance and resource constraints.

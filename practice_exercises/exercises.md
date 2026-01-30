# Practice Exercises for Advanced Machine Learning

This document contains practice exercises to reinforce concepts from the Advanced Machine Learning course (SDS 6217).

---

## Exercise 1: Linear Regression Fundamentals

**Objective**: Implement linear regression from scratch using gradient descent.

**Tasks**:

1. Generate a synthetic dataset with 100 points following the equation y = 2x + 1 + noise
2. Implement the linear regression model with:
   - Weights (w) and bias (b) initialization
   - MSE loss function
   - Gradient descent optimization
3. Train the model for 1000 iterations with learning rate 0.1
4. Plot the loss curve over iterations
5. Compare learned parameters with true parameters

**Questions to Answer**:

- What happens if the learning rate is too high (e.g., 1.0)?
- What happens if the learning rate is too low (e.g., 0.0001)?
- Why do only the weights and bias get updated during training?

---

## Exercise 2: Classification Metrics Calculations

**Objective**: Practice calculating and interpreting classification metrics.

**Given Confusion Matrix**:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | 45 | 15 |
| **Actual Negative** | 10 | 30 |

**Tasks**:

1. Calculate the following metrics:
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - Specificity
   - F1 Score
   - False Positive Rate
   - False Negative Rate

2. Interpret what each metric tells us about the model performance

3. If we want to reduce false positives, what should we do to the threshold?

4. If we want to improve recall, what should we do to the threshold?

**Bonus Challenge**:

Consider the insect detection scenario from the quiz:
- Invasive species are rare (5% of data)
- Missing an invasive species is very costly
- False alarms are easy to handle

Design a classification strategy that prioritizes catching all invasive species, even at the cost of some false alarms.

---

## Exercise 3: Neural Network Components

**Objective**: Understand the components and operations of neural networks.

**Tasks**:

1. Create a simple neural network with architecture [4, 3, 2, 1]
2. Implement forward propagation manually:
   - Linear transformation: z = x @ w + b
   - Activation function (use ReLU for hidden layers, Sigmoid for output)
3. Trace the dimensions of data through each layer
4. Implement backpropagation for gradient computation
5. Update weights using gradient descent

**Questions to Answer**:

- What is the purpose of activation functions?
- Why do we need non-linear activation functions?
- What happens during forward propagation vs. backpropagation?
- How does the learning rate affect training?

---

## Exercise 4: CNN vs. Other Architectures

**Objective**: Understand when to use different neural network architectures.

**Tasks**:

1. Create a simple image (8x8) with a square pattern
2. Apply convolution with edge detection kernels
3. Apply max pooling to the convolved features
4. Compare with MLP predictions for the same task

**Questions to Answer**:

- Why are CNNs better suited for image processing than MLPs?
- What is the purpose of pooling layers?
- How do CNNs achieve translation invariance?
- When would you use RNNs/LSTMs instead of CNNs?

---

## Exercise 5: Clustering Algorithms

**Objective**: Implement and compare clustering algorithms.

**Tasks**:

1. Generate three types of cluster structures:
   - Spherical clusters (using make_blobs)
   - Elongated clusters
   - Non-convex clusters (using make_moons)
2. Apply K-Means clustering to each
3. Visualize results and identify limitations
4. Implement the elbow method to find optimal k

**Questions to Answer**:

- What are the assumptions of K-Means clustering?
- Why does K-Means fail on non-convex clusters?
- What is the difference between partitional and hierarchical clustering?
- How do you determine the optimal number of clusters?

---

## Exercise 6: Data Preparation and Splitting

**Objective**: Understand best practices for data preparation.

**Tasks**:

1. Generate a dataset of 1000 samples with 10 features
2. Create a proper train/validation/test split:
   - 70% training, 15% validation, 15% test
   - Ensure stratified sampling for imbalanced classes
3. Implement proper preprocessing:
   - Standard scaling (fit on training, transform all)
   - PCA for dimensionality reduction
4. Detect and handle data leakage

**Questions to Answer**:

- Why should we fit preprocessing on training data only?
- What is data leakage and how to prevent it?
- How does the size of test set affect our confidence in results?
- What is the purpose of a validation set?

---

## Exercise 7: Imbalanced Datasets

**Objective**: Handle highly imbalanced datasets.

**Scenario**:

- Training set: 1 billion majority class, 10 million minority class
- Imbalance ratio: 100:1
- Batch size: 128

**Tasks**:

1. Calculate expected minority class samples per batch
2. Implement strategies to handle imbalance:
   - Downsampling majority class
   - Upsampling minority class
   - Class weights in loss function
3. Compare model performance with different strategies

**Questions to Answer**:

- Why does imbalance cause problems for training?
- What is the recommended ratio after downsampling?
- Why not just use a very large batch size?
- What other techniques can handle imbalance?

---

## Exercise 8: Generalization and Overfitting

**Objective**: Understand and prevent overfitting.

**Tasks**:

1. Generate a dataset with clear patterns
2. Train models with different complexities:
   - Simple linear model
   - Polynomial features (degree 2, 5, 10)
   - Neural network with different hidden layer sizes
3. Evaluate on training and test sets
4. Plot learning curves

**Questions to Answer**:

- How does model complexity relate to overfitting?
- What signs indicate overfitting?
- How does regularization prevent overfitting?
- Why might test loss be suspiciously low?

---

## Exercise 9: Reinforcement Learning Concepts

**Objective**: Understand reinforcement learning paradigm.

**Tasks**:

1. Implement a simple environment (e.g., grid world)
2. Create an agent that learns through exploration
3. Implement Q-learning algorithm
4. Visualize learning progress over episodes

**Questions to Answer**:

- How does reinforcement learning differ from supervised learning?
- What is the role of the reward function?
- What is the exploration vs. exploitation trade-off?
- How do we define a "policy" in RL?

---

## Exercise 10: Feature Selection

**Objective**: Practice feature selection strategies.

**Tasks**:

1. Generate a dataset with 20 features, only 3 truly predictive
2. Implement forward feature selection
3. Compare with correlation-based selection
4. Evaluate model performance with different feature sets

**Questions to Answer**:

- Why start with 1-3 features instead of many?
- What are the risks of using too many features?
- How does feature selection prevent overfitting?
- What makes a feature "predictive"?

---

## Answer Key

### Exercise 1 Answers

1. **Learning rate too high**: Model may diverge or oscillate
2. **Learning rate too low**: Model converges very slowly
3. **Parameters updated**: Only weights and bias (not feature values or predictions)

### Exercise 2 Answers

Given confusion matrix:
- TP = 45, TN = 30, FP = 10, FN = 15
- Accuracy = (45 + 30) / 100 = 75%
- Precision = 45 / (45 + 10) = 81.82%
- Recall = 45 / (45 + 15) = 75%
- F1 = 2 * 0.8182 * 0.75 / (0.8182 + 0.75) = 78.26%
- Specificity = 30 / (30 + 10) = 75%

### Exercise 3 Answers

1. Activation functions introduce non-linearity, enabling the network to learn complex patterns
2. Non-linear activations are needed to approximate any function (universal approximation)
3. Forward propagation computes predictions; backpropagation computes gradients
4. Learning rate controls step size in gradient descent

### Exercise 4 Answers

1. CNNs preserve spatial relationships, use shared weights, detect local features
2. Pooling reduces dimensions and provides translation invariance
3. CNNs work well because images have local patterns that can be detected regardless of position

### Exercise 5 Answers

1. K-Means assumes spherical clusters of similar size
2. K-Means fails on non-convex shapes because it uses distance to centroids
3. Partitional creates flat clusters; hierarchical creates tree structure
4. Elbow method, silhouette score, domain knowledge help determine k

### Exercise 6 Answers

1. Fitting on training only prevents data leakage from test/validation
2. Data leakage: information from future leaking into training; prevent by proper splitting
3. Larger test set = more statistically significant results

### Exercise 7 Answers

1. With batch size 128 and 100:1 ratio, expect about 1-2 minority samples per batch
2. Recommended: downsample to 20:1 ratio and upweight minority
3. Larger batch size makes minority representation worse

### Exercise 8 Answers

1. Higher complexity = more overfitting risk
2. Signs: Training loss low, test loss high; large gap between train and test metrics
3. Regularization adds penalty for large weights

### Exercise 9 Answers

1. RL uses rewards instead of labels; learns from interaction
2. Reward function defines what the agent should achieve
3. Exploration: trying new actions; exploitation: using known good actions

### Exercise 10 Answers

1. Start simple to avoid overfitting, ensure interpretability
2. Too many features: overfitting, slower training, harder interpretation
3. Predictive features have strong correlation with target and causal relationship

---

## Additional Resources

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **Fast.ai Courses**: https://www.fast.ai/
- **Coursera ML Specialization**: Andrew Ng's courses

---

## Author

**Cavin Otieno Ouma**
- Registration: SDS6/46982/2024
- Programme: MSc Public Health Data Science
- Institution: University of Nairobi, Department of Mathematics

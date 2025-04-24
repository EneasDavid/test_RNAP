import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Import the student implementation
from student_template import train_perceptron
# from solution import train_perceptron

# Generate blobs and modify to make them non-linearly separable
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=4.5, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert to -1 and 1

# Shuffle the data
rng = np.random.RandomState(42)
indices = rng.permutation(len(X))
X = X[indices]
y = y[indices]

# Add bias term to features
X_bias = np.c_[np.ones(X.shape[0]), X]

# Train perceptron
weights = train_perceptron(X_bias, y, epochs=50, lr=0.1)

# Predict
preds = np.sign(np.dot(X_bias, weights))

# Calculate accuracy
accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy:.2f}")
# Expected: 0.72

# Plot decision boundary
def plot_decision_boundary(X, y, weights):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='bwr', edgecolors='k')

    # Compute decision boundary
    x_vals = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
    plt.plot(x_vals, y_vals, '--k')

    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_bias, y, weights)

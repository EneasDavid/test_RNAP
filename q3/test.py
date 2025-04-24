import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import student function
from student_template import build_and_train_model
# from solution import build_and_train_model

# Generate dataset
np.random.seed(0)
x = np.linspace(0, 1, 300)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, x.shape)

X = x.reshape(-1, 1)
y = y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get student predictions
y_pred = build_and_train_model(X_train, y_train, X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='lightblue', label='Train Data', alpha=0.6)
plt.scatter(X_test, y_test, color='gray', label='Test Data', alpha=0.6)
plt.scatter(X_test, y_pred, color='red', label='Predictions', s=10)
plt.title("Regularized DNN Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

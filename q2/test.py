import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Import the student's implementation
from student_template import build_and_train_model
# from solution import build_and_train_model

# Simulate dataset
def generate_data():
    np.random.seed(42)
    n_samples = 500

    numeric_data = np.random.randn(n_samples, 3)
    categorical_data = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))

    X = pd.DataFrame(np.hstack([numeric_data, categorical_data]), columns=['num1', 'num2', 'num3', 'cat1', 'cat2'])
    X[['num1', 'num2', 'num3']] = X[['num1', 'num2', 'num3']].astype(float)
    X[['cat1', 'cat2']] = X[['cat1', 'cat2']].astype(str)

    y = 3 * X['num1'].astype(float) - 2 * X['num2'].astype(float) + X['num3'].astype(float) + \
        (X['cat1'] == 'A').astype(float) * 1.5 + (X['cat2'] == 'C').astype(float) * 2 + np.random.randn(n_samples)

    return X, y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run student model
y_pred = build_and_train_model(X_train, y_train, X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Expected:
# Test MSE: 1.0822
# Test MSE: 1.2192
# Test MSE: 1.2132
# Test MSE: 1.2634
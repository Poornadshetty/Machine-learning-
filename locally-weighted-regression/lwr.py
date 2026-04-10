import numpy as np
import matplotlib.pyplot as plt

# Compute weight matrix
def weight_matrix(x_query, X, tau):
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau ** 2))
    return np.diag(weights)

# Locally Weighted Regression function
def locally_weighted_regression(X, Y, tau, x_query):
    X = np.c_[np.ones(X.shape[0]), X]   # Add bias term
    x_query = np.r_[1, x_query]         # Add bias to query point

    W = weight_matrix(x_query, X, tau)  # Weight matrix

    # Normal equation with weights
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y

    # Prediction
    y_pred = x_query @ theta
    return y_pred

# Generate dataset
X = np.linspace(-5, 5, 200).reshape(-1, 1)
Y = np.sin(X).flatten() + np.random.normal(0, 0.1, 200)

# Bandwidth parameter
tau = 0.5

# Predictions
x_range = np.linspace(-5, 5, 200)
y_pred = [locally_weighted_regression(X, Y, tau, np.array([x])) for x in x_range]

# Plot
plt.scatter(X, Y, color='red', label='Data Points')
plt.plot(x_range, y_pred, color='black', label='LWR Prediction')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

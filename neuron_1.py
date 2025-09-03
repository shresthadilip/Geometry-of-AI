import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Generate toy dataset
np.random.seed(0)
N = 100
X_class0 = np.random.randn(N, 2) - 2  # cluster around (-2, -2)
X_class1 = np.random.randn(N, 2) + 2  # cluster around (+2, +2)
X = np.vstack([X_class0, X_class1])
y = np.hstack([np.zeros(N), np.ones(N)]).reshape(-1, 1)

# Add bias term (x1, x2, bias)
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# 2. Logistic regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    return sigmoid(np.dot(X, w))

def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# 3. Training loop (store weights at each step for animation)
w = np.random.randn(3, 1)  # weights for x1, x2, bias
lr = 0.1
epochs = 50
weights_history = [w.copy()]

for epoch in range(epochs):
    y_hat = predict(X_bias, w)
    gradient = np.dot(X_bias.T, (y_hat - y)) / y.shape[0]
    w -= lr * gradient
    weights_history.append(w.copy())

# 4. Animation of decision boundary
fig, ax = plt.subplots()
ax.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0")
ax.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1")
line, = ax.plot([], [], 'k-', linewidth=2)
ax.legend()

def update(i):
    w = weights_history[i]
    x_vals = np.array(ax.get_xlim())
    y_vals = -(w[0] * x_vals + w[2]) / w[1]
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {i}")
    return line,

ani = FuncAnimation(fig, update, frames=len(weights_history), interval=300, repeat=False)
plt.show()

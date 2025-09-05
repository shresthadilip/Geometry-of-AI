import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset_factory import generate_linearly_separable_data

# --- Generate toy dataset ---
N = 100
X, y, X_class0, X_class1 = generate_linearly_separable_data(N=N, seed=0)

# Add bias
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Predictions
def predict(X, w):
    return sigmoid(np.dot(X, w))

# Loss
def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# --- Training ---
w = np.random.randn(3, 1)
lr = 0.1
epochs = 100
weights_history, losses = [], []

for epoch in range(epochs):
    y_hat = predict(X_bias, w)
    gradient = np.dot(X_bias.T, (y_hat - y)) / y.shape[0]
    w -= lr * gradient
    weights_history.append(w.copy())
    losses.append(compute_loss(y, y_hat))

# --- Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left plot: decision boundary
ax1.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0")
ax1.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1")
line, = ax1.plot([], [], 'k-', linewidth=2)
ax1.legend()
ax1.set_title("Decision Boundary")

# Right plot: loss curve
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, max(losses) * 1.1)
ax2.set_title("Loss Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
loss_line, = ax2.plot([], [], 'r-')

def update(i):
    # Decision boundary
    w = weights_history[i]
    x_vals = np.array(ax1.get_xlim())
    y_vals = -(w[0] * x_vals + w[2]) / w[1]
    line.set_data(x_vals, y_vals)
    
    # Loss curve
    loss_line.set_data(range(i+1), losses[:i+1])
    
    return line, loss_line

ani = FuncAnimation(fig, update, frames=len(weights_history), interval=300, repeat=False)
plt.show()

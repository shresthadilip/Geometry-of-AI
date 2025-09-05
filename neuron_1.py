import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset_factory import generate_non_linearly_separable_data

# 1. Generate toy dataset
N = 100
X, y, X_class0, X_class1 = generate_non_linearly_separable_data(N=N, seed=0)

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
epochs = 200
weights_history = [w.copy()]

for epoch in range(epochs):
    y_hat = predict(X_bias, w)
    gradient = np.dot(X_bias.T, (y_hat - y)) / y.shape[0]
    w -= lr * gradient
    weights_history.append(w.copy())

# 4. Animation of decision boundary
fig, ax = plt.subplots()
ax.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0", alpha=0.7)
ax.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1", alpha=0.7)
line, = ax.plot([], [], 'k-', linewidth=2)

ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.legend()

def update(i):
    w = weights_history[i]
    x_vals = np.array(ax.get_xlim())
    # Add a small epsilon to avoid division by zero for vertical lines
    y_vals = -(w[0] * x_vals + w[2]) / (w[1] + 1e-8)
    line.set_data(x_vals, y_vals)
    ax.set_title(f"Epoch {i+1}")
    return line,

ani = FuncAnimation(fig, update, frames=len(weights_history), interval=300, repeat=False)

# To save the animation as a GIF, you might need to install Pillow:
# pip install Pillow
output_filename = 'neuron_1_animation.gif'
print(f"Saving animation to {output_filename}...")
ani.save(output_filename, writer='pillow', fps=5)
print("Done saving GIF.")
# plt.show() # You can uncomment this if you want to see the plot after saving

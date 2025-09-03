import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Toy dataset: XOR pattern (not linearly separable)
np.random.seed(1)
N = 100
X = np.random.randn(N, 2)
y = (X[:, 0] * X[:, 1] > 0).astype(int).reshape(-1, 1)  # XOR-like labels

# Add bias term
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# Activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Neural net forward pass
def forward(X, W1, W2):
    z1 = X @ W1      # hidden layer (2 neurons)
    a1 = sigmoid(z1)
    z2 = np.hstack([a1, np.ones((a1.shape[0], 1))]) @ W2  # add bias for output
    a2 = sigmoid(z2)
    return a1, a2

# Loss function
def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# Initialize weights
W1 = np.random.randn(3, 2) * 0.5  # input->hidden (2 neurons)
W2 = np.random.randn(3, 1) * 0.5  # hidden->output

lr = 0.1
epochs = 60
history = []

# Training loop (simple gradient descent, no backprop libs)
for epoch in range(epochs):
    # Forward
    a1, y_hat = forward(X_bias, W1, W2)

    # Backpropagation
    error2 = y_hat - y
    grad_W2 = np.hstack([a1, np.ones((a1.shape[0], 1))]).T @ error2 / N

    error1 = (error2 @ W2[:-1].T) * a1 * (1 - a1)
    grad_W1 = X_bias.T @ error1 / N

    # Update weights
    W1 -= lr * grad_W1
    W2 -= lr * grad_W2

    # Save state for animation
    history.append((W1.copy(), W2.copy()))

# --- Visualization ---
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", alpha=0.7)
line1, = ax.plot([], [], 'g--', label="Neuron 1")
line2, = ax.plot([], [], 'm--', label="Neuron 2")
ax.legend()

def update(frame):
    W1, W2 = history[frame]

    # Plot hidden neuron decision boundaries
    x_vals = np.array(ax.get_xlim())
    # neuron1
    y1_vals = -(W1[0,0]*x_vals + W1[2,0]) / (W1[1,0]+1e-6)
    # neuron2
    y2_vals = -(W1[0,1]*x_vals + W1[2,1]) / (W1[1,1]+1e-6)

    line1.set_data(x_vals, y1_vals)
    line2.set_data(x_vals, y2_vals)
    ax.set_title(f"Epoch {frame+1}")
    return line1, line2

ani = FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
plt.show()

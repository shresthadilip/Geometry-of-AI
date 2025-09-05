import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset_factory import (
    generate_linearly_separable_data,
    generate_non_linearly_separable_data,
    generate_circular_dataset
)

# -------------------
# Dataset
# -------------------
N = 100
# Using the non-linear dataset as it's a good test for multi-neuron networks
X, y, X_class0, X_class1 = generate_non_linearly_separable_data(N=N, seed=0)

X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# -------------------
# Helpers
# -------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, W2):
    a1 = sigmoid(X @ W1)
    z2 = np.hstack([a1, np.ones((a1.shape[0],1))]) @ W2
    y_hat = sigmoid(z2)
    return a1, y_hat

def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat+1e-8) + (1-y)*np.log(1-y_hat+1e-8))

# -------------------
# Training + Animation function
# -------------------
def train_and_visualize(neurons, epochs=60, lr=0.1):
    W1 = np.random.randn(3, neurons) * 0.5
    W2 = np.random.randn(neurons+1, 1) * 0.5
    history_W1, history_W2, losses = [], [], []

    for epoch in range(epochs):
        a1, y_hat = forward(X_bias, W1, W2)
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # Backprop
        error2 = y_hat - y
        grad_W2 = np.hstack([a1, np.ones((a1.shape[0],1))]).T @ error2 / X.shape[0]
        error1 = (error2 @ W2[:-1].T) * a1 * (1-a1)
        grad_W1 = X_bias.T @ error1 / X.shape[0]

        W1 -= lr * grad_W1
        W2 -= lr * grad_W2

        history_W1.append(W1.copy())
        history_W2.append(W2.copy())

    # --- Setup plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0")
    ax1.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1")
    ax1.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
    ax1.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
    ax1.set_title(f"{neurons} Hidden Neurons")
    ax1.legend(loc='upper left')
    neuron_lines = []

    ax2.set_xlim(0, epochs)
    ax2.set_ylim(0, max(losses)*1.1)
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    loss_line, = ax2.plot([], [], 'r-')

    xs = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)

    def update(frame):
        nonlocal neuron_lines
        for ln in neuron_lines:
            ln.remove()
        neuron_lines = []

        W1 = history_W1[frame]
        for j in range(neurons):
            w = W1[:,j]
            if abs(w[1]) > 1e-6:
                ys = -(w[0]*xs + w[2])/w[1]
                ln, = ax1.plot(xs, ys, '--k', alpha=0.5)
                neuron_lines.append(ln)

        loss_line.set_data(range(frame+1), losses[:frame+1])
        ax2.set_title(f"Loss Curve (Epoch {frame+1})")
        return *neuron_lines, loss_line

    ani = FuncAnimation(fig, update, frames=epochs, interval=200, repeat=False)
    plt.tight_layout()
    plt.show()

# -------------------
# Run sequentially for 3, 4, 5 neurons
# -------------------
for n in [1, 2, 3, 5, 8]:
    train_and_visualize(neurons=n, epochs=100)

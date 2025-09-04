import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------
# Dataset
# -------------------
np.random.seed(42)
N = 500
X = np.random.uniform(0, 14, (N, 2))   # spread points in [0,14]x[0,14]
y = (np.linalg.norm(X - np.array([7,7]), axis=1) > 4).astype(int).reshape(-1, 1)

# Add bias
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# -------------------
# Helpers
# -------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, W2):
    z1 = X @ W1
    a1 = sigmoid(z1)
    z2 = np.hstack([a1, np.ones((a1.shape[0], 1))]) @ W2
    y_hat = sigmoid(z2)
    return a1, y_hat

def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# -------------------
# 2-Layer Training + Animation
# -------------------
def train_and_animate_2layer(neurons=5, epochs=100, lr=0.1):
    # W1: input (2+1) -> hidden layer
    W1 = np.random.randn(3, neurons) * 0.5
    # W2: hidden layer (+bias) -> output
    W2 = np.random.randn(neurons+1, 1) * 0.5
    losses = []
    history_W1, history_W2 = [], []

    for epoch in range(epochs):
        a1, y_hat = forward(X_bias, W1, W2)
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # --- Backprop ---
        error2 = y_hat - y
        grad_W2 = np.hstack([a1, np.ones((a1.shape[0],1))]).T @ error2 / N
        error1 = (error2 @ W2[:-1].T) * a1 * (1 - a1)
        grad_W1 = X_bias.T @ error1 / N

        W1 -= lr * grad_W1
        W2 -= lr * grad_W2

        history_W1.append(W1.copy())
        history_W2.append(W2.copy())

    # --- Setup plots side by side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    scatter = ax1.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="bwr", edgecolor="k", alpha=0.7)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 14)
    ax1.set_title(f"2-Layer Network Decision Boundary ({neurons} neurons)")

    xx, yy = np.meshgrid(np.linspace(0,14,200), np.linspace(0,14,200))
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]
    contour = None
    line_artists = []

    ax2.set_xlim(0, epochs)
    ax2.set_ylim(0, 2*max(losses))
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    loss_line, = ax2.plot([], [], 'r-')

    # --- Animation ---
    def update(frame):
        nonlocal contour, line_artists
        if contour is not None:
            try:
                contour.remove()
            except Exception:
                pass
        for line in line_artists:
            line.remove()
        line_artists = []

        W1_f = history_W1[frame]
        W2_f = history_W2[frame]
        _, y_grid = forward(grid, W1_f, W2_f)
        y_grid = y_grid.reshape(xx.shape)
        contour = ax1.contourf(xx, yy, y_grid, levels=[0,0.5,1], alpha=0.5, cmap="bwr")

        # --- Plot hidden neuron lines ---
        x_vals = np.linspace(0, 14, 200)
        for i in range(neurons):
            w = W1_f[:, i]
            if abs(w[1]) > 1e-6:  # avoid division by zero
                y_vals = -(w[0] * x_vals + w[2]) / w[1]
                line, = ax1.plot(x_vals, y_vals, 'k--', alpha=0.6)
                line_artists.append(line)

        # Loss curve
        loss_line.set_data(range(frame+1), losses[:frame+1])
        ax2.set_title(f"Loss Curve (Epoch {frame+1})")
        return [contour, loss_line] + line_artists

    ani = FuncAnimation(fig, update, frames=epochs, interval=100, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

# -------------------
# Run for different neuron counts
# -------------------
for n in [3,4,5]:
    train_and_animate_2layer(neurons=n)

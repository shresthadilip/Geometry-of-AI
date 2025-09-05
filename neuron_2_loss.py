import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset_factory import generate_non_linearly_separable_data

# --- Dataset: Three clusters (not linearly separable) ---
N = 100
X, y, X_class0, X_class1 = generate_non_linearly_separable_data(N=N, seed=0)
X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

# --- Activation ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward pass
def forward(X, W1, W2):
    z1 = X @ W1
    a1 = sigmoid(z1)
    z2 = np.hstack([a1, np.ones((a1.shape[0], 1))]) @ W2
    a2 = sigmoid(z2)
    return a1, a2

# Loss
def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

# --- Initialize weights ---
W1 = np.random.randn(3, 2) * 0.5
W2 = np.random.randn(3, 1) * 0.5
lr = 0.1
epochs = 60
history, losses = [], []

# --- Training ---
for epoch in range(epochs):
    a1, y_hat = forward(X_bias, W1, W2)
    loss = compute_loss(y, y_hat)
    
    # Backprop
    error2 = y_hat - y
    grad_W2 = np.hstack([a1, np.ones((a1.shape[0], 1))]).T @ error2 / X.shape[0]
    
    error1 = (error2 @ W2[:-1].T) * a1 * (1 - a1)
    grad_W1 = X_bias.T @ error1 / X.shape[0]
    
    W1 -= lr * grad_W1
    W2 -= lr * grad_W2
    
    history.append((W1.copy(), W2.copy()))
    losses.append(loss)

# --- Animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left: dataset + neuron boundaries
ax1.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0", alpha=0.7)
ax1.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1", alpha=0.7)
ax1.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

line1, = ax1.plot([], [], 'g--', label="Neuron 1")
line2, = ax1.plot([], [], 'm--', label="Neuron 2")
ax1.legend()
ax1.set_title("Hidden Neuron Lines")

# Right: loss curve
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, max(losses) * 1.1)
ax2.set_title("Loss Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
loss_line, = ax2.plot([], [], 'r-')

def update(frame):
    W1, W2 = history[frame]
    x_vals = np.array(ax1.get_xlim())
    
    # neuron 1
    y1_vals = -(W1[0,0]*x_vals + W1[2,0]) / (W1[1,0]+1e-6)
    line1.set_data(x_vals, y1_vals)
    
    # neuron 2
    y2_vals = -(W1[0,1]*x_vals + W1[2,1]) / (W1[1,1]+1e-6)
    line2.set_data(x_vals, y2_vals)
    
    # loss curve
    loss_line.set_data(range(frame+1), losses[:frame+1])
    
    return line1, line2, loss_line

ani = FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
plt.show()

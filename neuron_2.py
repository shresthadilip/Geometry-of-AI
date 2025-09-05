import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataset_factory import generate_non_linearly_separable_data

# 1. Toy dataset: Three clusters (not linearly separable)
N = 100
X, y, X_class0, X_class1 = generate_non_linearly_separable_data(N=N, seed=0)

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
# We manually set the initial weights for W1 to create a more visually
# interesting starting point for the animation.
# Neuron 1 starts as a horizontal line, Neuron 2 as a vertical line.
W1 = np.array([
    [0.0, 1.0],  # weights for x1
    [1.0, 0.0],  # weights for x2
    [-3.0, -3.0] # bias weights
], dtype=float)
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
    grad_W2 = np.hstack([a1, np.ones((a1.shape[0], 1))]).T @ error2 / X.shape[0]

    error1 = (error2 @ W2[:-1].T) * a1 * (1 - a1)
    grad_W1 = X_bias.T @ error1 / X.shape[0]

    # Update weights
    W1 -= lr * grad_W1
    W2 -= lr * grad_W2

    # Save state for animation
    history.append((W1.copy(), W2.copy()))

# --- Visualization ---
fig, ax = plt.subplots()
ax.scatter(X_class0[:, 0], X_class0[:, 1], color="red", label="Class 0", alpha=0.7)
ax.scatter(X_class1[:, 0], X_class1[:, 1], color="blue", label="Class 1", alpha=0.7)

ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

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
    ax.set_title(f"Epoch {frame}")
    return line1, line2

ani = FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)

# plt.show()

# To save the animation as a GIF, you might need to install Pillow:
# pip install Pillow
output_filename = 'neuron_2_animation.gif'
print(f"Saving animation to {output_filename}...")
ani.save(output_filename, writer='pillow', fps=5)
print("Done saving GIF.")
# plt.show() # You can uncomment this if you want to see the plot after saving

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------
# Function 2: Circular dataset
# -------------------
def generate_circular_dataset(N=500, center=(7,7), radius=4, seed=42):
    """
    Generates dataset with points inside and outside a circle.
    Class 0 = inside radius, Class 1 = outside radius
    Returns X (points) and y (labels)
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 14, (N, 2))
    y = (np.linalg.norm(X - np.array(center), axis=1) > radius).astype(int).reshape(-1, 1)
    return X, y

# -------------------
# Helpers
# -------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat+1e-8) + (1-y)*np.log(1-y_hat+1e-8))

# -------------------
# Training + Animation for CIRCULAR DATASET
# -------------------
def train_and_visualize_circular(epochs=500, lr=0.1, center=(7,7)):
    
    # Generate the circular dataset
    X, y = generate_circular_dataset(N=500, center=center)
    
    # Feature Engineering: Add a new feature representing squared distance from the center.
    # This transforms the problem from a non-linear 2D problem to a linear 3D one.
    dist_sq = np.sum((X - np.array(center))**2, axis=1, keepdims=True)
    X_enhanced = np.hstack([X, dist_sq])
    
    # Add bias term for the linear model
    X_bias = np.hstack([X_enhanced, np.ones((X_enhanced.shape[0], 1))])
    
    # Initialize weights for a simple perceptron (no hidden layers needed now)
    W = np.random.randn(4, 1) * 0.5
    
    history_W, losses = [], []

    for epoch in range(epochs):
        # Forward Pass
        y_hat = sigmoid(X_bias @ W)
        loss = compute_loss(y, y_hat)
        losses.append(loss)

        # Backpropagation
        error = y_hat - y
        grad_W = X_bias.T @ error / X.shape[0]
        W -= lr * grad_W

        history_W.append(W.copy())
        
    # --- Setup plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr")
    ax1.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
    ax1.set_ylim(X[:,1].min()-1, X[:,1].max()+1)
    ax1.set_title("Circular Decision Boundary")
    ax1.legend()
    
    ax2.set_xlim(0, epochs)
    ax2.set_ylim(0, max(losses)*1.1)
    ax2.set_title("Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    loss_line, = ax2.plot([], [], 'r-')
    
    # --- Animation function ---
    def update(frame):
        # Get weights from history
        W = history_W[frame]
        
        # Calculate the boundary circle from the weights
        # The equation for the decision boundary is a*x + b*y + c*(x-cx)^2 + c*(y-cy)^2 + d = 0
        # where (cx, cy) is the chosen center, and the weights W = [a, b, c, d].
        a, b, c, d = W.ravel()
        
        # We solve for the radius, which is sqrt(-d/c) if the center is at (0,0) and a and b are 0.
        # With the added features, the boundary is a linear combination of x, y, and (x-cx)^2+(y-cy)^2.
        # This linear combination of features in the 3D space projects to a circle in the 2D space.
        # The radius is given by the formula for a circle.
        if 'boundary_circle' in update.__dict__:
            update.boundary_circle.remove()

        try:
            radius = np.sqrt(-d / c)
            update.boundary_circle = plt.Circle(center, radius, fill=False, color='green', linewidth=2)
            ax1.add_patch(update.boundary_circle)
        except (ValueError, ZeroDivisionError):
            pass # Handle cases where radius is invalid

        loss_line.set_data(range(frame+1), losses[:frame+1])
        ax2.set_title(f"Loss Curve (Epoch {frame+1})")
        
        return update.boundary_circle, loss_line
        
    ani = FuncAnimation(fig, update, frames=epochs, interval=200, repeat=False)
    plt.tight_layout()
    plt.show()

# Run the new function for circular data
train_and_visualize_circular()

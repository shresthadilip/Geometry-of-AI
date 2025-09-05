import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Layer configuration: number of nodes per layer
layers = [1, 13, 3, 9, 4, 8, 5, 9, 11]

# Node positions
pos = {}
node_idx = 0
layer_spacing = 8
node_spacing = 4
for i, n_nodes in enumerate(layers):
    y_positions = np.linspace(-(n_nodes-1)/2, (n_nodes-1)/2, n_nodes) * node_spacing
    for y in y_positions:
        pos[node_idx] = (i * layer_spacing, y)
        node_idx += 1

# Build edges (fully connected between consecutive layers)
edges = []
node_idx = 0
for i, n_nodes in enumerate(layers[:-1]):
    next_idx = node_idx + n_nodes
    for u in range(node_idx, node_idx + n_nodes):
        for v in range(next_idx, next_idx + layers[i+1]):
            edges.append((u, v))
    node_idx = next_idx

# Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor("#2d2d2d")
ax.set_facecolor("#2d2d2d")
ax.axis("off")
ax.set_aspect('equal')

# Draw nodes
for node, (x, y) in pos.items():
    circle = plt.Circle((x, y), 0.8,
                        edgecolor="#a5a5a7", facecolor="#212327", lw=1.2, zorder=3)
    ax.add_patch(circle)

# Draw edges
for u, v in edges:
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    ax.plot([x0, x1], [y0, y1], color="#a5a5a7", lw=0.3, alpha=0.6, zorder=1)

# Particle settings
n_particles = 180
particles = np.random.choice(len(edges), n_particles)
particle_pos = np.random.rand(n_particles)
scat = ax.scatter([], [], s=8, c="white", zorder=4)

###
# Animation update function
def update(frame):
    global particle_pos
    particle_pos = (particle_pos + 0.02) % 1.0

    x, y = [], []
    for i in range(n_particles):
        u, v = edges[particles[i]]
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # linear interpolation
        x.append(x0 + particle_pos[i] * (x1 - x0))
        y.append(y0 + particle_pos[i] * (y1 - y0))

    scat.set_offsets(np.c_[x, y])
    return scat,

ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# To save the animation as a GIF, you might need to install Pillow:
# pip install Pillow
output_filename = 'fun_animation.gif'
print(f"Saving animation to {output_filename}...")
# The interval is 50ms, so 1000/50 = 20 fps.
ani.save(output_filename, writer='pillow', fps=20)
print("Done saving GIF.")

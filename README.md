# Geometry of AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

An exploration into the geometric structures of neural networks and machine learning models. This repository contains code, notebooks, and visualizations for understanding the geometry of AI.

## 📖 Table of Contents

- [About The Project](#about-the-project)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🌟 About The Project

The "Geometry of AI" project aims to demystify the inner workings of machine learning models by studying their underlying geometric properties. By visualizing loss landscapes, analyzing the structure of representation spaces, and applying concepts from differential geometry and topology, we can gain deeper insights into why these models work so well and how they can be improved.

This repository serves as a collection of experiments and tools to explore these geometric perspectives.

## 🔑 Key Concepts

This project may explore several key areas at the intersection of geometry and AI, including:

*   **Loss Landscapes**: Visualizing and analyzing the high-dimensional error surfaces that models navigate during training.
*   **Representation Spaces**: Understanding the geometric structure of data embeddings and how models organize information.
*   **The Manifold Hypothesis**: The idea that high-dimensional data often lies on or near a low-dimensional manifold.
*   **Information Geometry**: Applying geometric methods to the space of probability distributions to understand learning dynamics.
*   **Topological Data Analysis (TDA)**: Using tools from topology to find robust structural features in data.
*   **Equivariance and Invariance**: How geometric symmetries in the data are handled by neural network architectures.

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

This project requires Python 3.8 or higher. You will also need `pip` for package management.

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Geometry-of-AI.git
    cd Geometry-of-AI
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

This section explains how to run the different Python programs included in this project. Based on the project's history, the scripts demonstrate various concepts in the geometry of neural networks, from single neurons to complex boundary learning.

### 1. Single Neuron Visualization

These scripts demonstrate how a single neuron, equivalent to a logistic regression model, learns to classify a linearly separable dataset.

*   **`neuron_1.py`**: This script creates an animation of the decision boundary. It visualizes the geometric aspect of learning, showing how the neuron's separating line moves through the data space with each training epoch to correctly classify the two clusters.

*   **`neuron_1_loss.py`**: This script enhances the visualization by adding a second, synchronized plot. It shows the decision boundary evolving on the left and the corresponding value of the loss function decreasing on the right. This provides a powerful intuition for how the geometric task of finding a good separating line is driven by the numerical process of minimizing a loss function.

**Example Commands:**

To see the decision boundary animation:
```bash
python neuron_1.py
```

To see the decision boundary and loss curve animated side-by-side:
```bash
python neuron_1_loss.py
```

### 2. Multi-Neuron Single Layer Networks

These scripts move beyond a single neuron to demonstrate how a hidden layer of neurons can learn to solve more complex problems by creating non-linear decision boundaries.

*   **`neuron_2.py` & `neuron_2_loss.py`**: These scripts tackle the classic **XOR problem**, which is not linearly separable. They train a network with a hidden layer of two neurons. The animation visualizes the individual decision boundaries of these two hidden neurons. You can see how the network learns to position these two lines to partition the data space, allowing the final output neuron to combine their results and correctly classify the XOR pattern. `neuron_2_loss.py` adds a synchronized loss plot.

*   **`neuron_3_loss.py`**: This script visualizes a network with **three hidden neurons** solving a linearly separable problem. It demonstrates how a network with more capacity than necessary can still find a valid, albeit sometimes more complex, solution by arranging its three linear boundaries.

*   **`neuron_x_loss.py`**: This is a configurable script that generalizes the previous examples. It runs a series of experiments, training and visualizing networks with 1, 2, 3, 4, and 5 hidden neurons sequentially. It's designed to explore the direct relationship between the number of neurons (model capacity) and the complexity of the decision boundary the network can learn. The script also includes code to easily switch to other datasets, like a circular one.

**Example Commands:**

```bash
# See how two neurons solve the XOR problem
python neuron_2.py

# See the two neurons and the loss curve for the XOR problem
python neuron_2_loss.py

# See how a variable number of neurons (1 through 5) learn a boundary
python neuron_x_loss.py
```
### 3. Multi-Layer Networks
This script provides the most complete picture. It trains a 2-layer network on a circular dataset and visualizes the final, non-linear decision boundary formed by the entire network. It does this by shading the regions of the plot according to the network's output. Crucially, it also overlays the individual linear boundaries of the hidden neurons. This powerfully demonstrates how the network combines simple linear separators to construct a complex, curved decision surface capable of solving non-linear problems. The script runs this visualization for networks with 3, 4, and 5 neurons, showing how model capacity impacts the final learned shape. 
**Example Command:**
```bash
python neuron_x_loss_multilayer.py
```

### 4. Nepal Map Boundary Learning

A fascinating experiment to train a neural network to learn a complex, non-convex shape like the map of Nepal. This script likely uses 2D points sampled from the map's boundary as training data and visualizes the network's attempt to replicate the shape.

**Example command (assuming `src/nepal_map_trainer.py`):**
```bash
python src/nepal_map_trainer.py --epochs 500
```

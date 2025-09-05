import numpy as np

def generate_linearly_separable_data(N=100, seed=0):
    """
    Generates three clusters. Two are class 0, one is class 1.
    The data is mostly linearly separable, suitable for a single neuron.
    """
    np.random.seed(seed)
    # Tighter clusters, further apart
    X_cluster1 = np.random.randn(N, 2) * 0.5 + np.array([2, 2])
    X_cluster2 = np.random.randn(N, 2) * 0.5 + np.array([4, 4])
    X_cluster3 = np.random.randn(N, 2) * 0.5 + np.array([6, 6])

    # Clusters around (2,2) and (4,4) are class 0
    X_class0 = np.vstack([X_cluster1, X_cluster2])
    # Cluster around (6,6) is class 1
    X_class1 = X_cluster3
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(2 * N), np.ones(N)]).reshape(-1, 1)
    
    return X, y, X_class0, X_class1

def generate_non_linearly_separable_data(N=100, seed=0):
    """
    Generates three clusters. The middle one is class 0, the outer two are class 1.
    This is not linearly separable, suitable for a multi-neuron network.
    """
    np.random.seed(seed)
    # Tighter clusters, further apart
    X_cluster1 = np.random.randn(N, 2) * 0.5 + np.array([2, 2])
    X_cluster2 = np.random.randn(N, 2) * 0.5 + np.array([4, 4])
    X_cluster3 = np.random.randn(N, 2) * 0.5 + np.array([6, 6])

    # Cluster around (4,4) is class 0
    X_class0 = X_cluster2
    # Clusters around (2,2) and (6,6) are class 1
    X_class1 = np.vstack([X_cluster1, X_cluster3])

    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(N), np.ones(2 * N)]).reshape(-1, 1)

    return X, y, X_class0, X_class1

def generate_circular_dataset(N=500, center=(7,7), radius=4, seed=42):
    """
    Generates dataset with points inside and outside a circle.
    Class 0 = inside radius, Class 1 = outside radius
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 14, (N, 2))
    y = (np.linalg.norm(X - np.array(center), axis=1) > radius).astype(int).reshape(-1, 1)
    return X, y
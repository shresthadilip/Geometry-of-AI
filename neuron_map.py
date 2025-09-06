import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import cv2
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset

def denorm(p, size=960):
    result = np.zeros_like(p)
    result[..., 0] = (p[..., 0] + 1) * (size / 2)
    result[..., 1] = size - ((p[..., 1] + 1) * (size / 2))
    return result

def load_and_process_map(map_path, color_a, color_b, threshold_a, threshold_b, image_size=960, num_samples=10000, seed=55):
    """
    Loads a map image, extracts coordinates for two colored regions,
    normalizes them, and creates full and sampled PyTorch tensors.

    Region from color_a will be labeled as 0.
    Region from color_b will be labeled as 1.

    Parameters
    ----------
    map_path : str
        Path to the map image file.
    color_a, color_b : np.array
        RGB colors for the two regions.
    threshold_a, threshold_b : int
        Color distance thresholds for the two regions.
    image_size : int, default: 960
        The size of the image (assuming a square image for normalization).
    num_samples : int, default: 10000
        Number of points to sample for each class.
    seed : int, default: 55
        Random seed for sampling.

    Returns
    -------
    tuple
        (X_tensor, y_tensor, X_sample, y_sample)
    """
    map_img = cv2.imread(map_path)[:, :, (2, 1, 0)]
    height, width, _ = map_img.shape

    region_a_mask = ((map_img - color_a)**2).sum(-1) < threshold_a
    region_b_mask = ((map_img - color_b)**2).sum(-1) < threshold_b

    coords_a_raw = np.array(np.where(region_a_mask)).T.astype('float')
    coords_b_raw = np.array(np.where(region_b_mask)).T.astype('float')

    # flip and normalize
    def normalize_coords(coords, w, h):
        norm_coords = np.zeros_like(coords)
        # coords are (row, col) which is (y, x)
        norm_coords[:, 0] = coords[:, 1] / (w / 2) - 1  # Normalize x with width
        norm_coords[:, 1] = (h - coords[:, 0]) / (h / 2) - 1  # Normalize y with height
        return norm_coords

    coords_a_all = normalize_coords(coords_a_raw, width, height)
    coords_b_all = normalize_coords(coords_b_raw, width, height)

    # Create full dataset tensors (class a: 0, class b: 1)
    X_full = np.vstack((coords_a_all, coords_b_all))
    y_full = np.concatenate((np.zeros(len(coords_a_all)),
                             np.ones(len(coords_b_all)))).astype('int')
    X_tensor = torch.FloatTensor(X_full)
    y_tensor = torch.tensor(y_full)

    # Create sampled dataset
    np.random.seed(seed)
    # Use replace=True if num_samples > len(coords)
    replace_a = num_samples > len(coords_a_all)
    sample_indices_a = np.random.choice(len(coords_a_all), num_samples, replace=replace_a)
    coords_a_sample = coords_a_all[sample_indices_a, :]

    replace_b = num_samples > len(coords_b_all)
    sample_indices_b = np.random.choice(len(coords_b_all), num_samples, replace=replace_b)
    coords_b_sample = coords_b_all[sample_indices_b, :]

    X_sample_np = np.vstack((coords_a_sample, coords_b_sample))
    y_sample_np = np.concatenate((np.zeros(len(coords_a_sample)),
                                  np.ones(len(coords_b_sample)))).astype('int')
    X_sample = torch.FloatTensor(X_sample_np)
    y_sample = torch.tensor(y_sample_np)

    return X_tensor, y_tensor, X_sample, y_sample, height, width

class BaarleNet(nn.Module):
    def __init__(self, hidden_layers=[64], activation='relu'):
        super(BaarleNet, self).__init__()
        
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(2, hidden_layers[0]), act_fn]
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, X_train, y_train, X_val, y_val, num_epochs=5, batch_size=10000, learning_rate=0.005, device='cpu'):
    """
    Trains a PyTorch model and prints progress.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    X_train, y_train : torch.Tensor
        Training data and labels.
    X_val, y_val : torch.Tensor
        Validation data and labels for accuracy checks.
    num_epochs : int, default: 5
        Number of training epochs.
    batch_size : int, default: 10000
        Size of mini-batches.
    learning_rate : float, default: 0.005
        Learning rate for the Adam optimizer.
    device : str, default: 'cpu'
        Device to train on ('cpu' or 'cuda').

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X.to(device))
            loss = criterion(outputs, batch_y.to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % (num_epochs // 5 or 1) == 0:
            with torch.no_grad():
                outputs_val = model(X_val.to(device))
                accuracy = (torch.argmax(outputs_val, dim=1) == y_val.to(device)).sum().item() / len(y_val)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return model

def viz_descision_boundary(model, height, width, map_path, device='cpu'):
    """
    Visualizes the model's decision boundary as an overlay on the map image.
    It probes the model on a grid of coordinates matching the image resolution
    to create a high-fidelity overlay.
    """
    dpi = 100.00
    # figsize is (width, height) in inches
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Create a grid with the same dimensions as the image for a high-res overlay.
    # mgrid is 'ij' indexing, so first dim is rows (y), second is cols (x).
    yy, xx = np.mgrid[-1:1:height*1j, -1:1:width*1j]
    # Stack the coordinates into (x, y) pairs, which the model expects.
    probe_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    # Get model predictions in batches to avoid memory errors with large images
    all_preds = []
    probe_tensor = torch.tensor(probe_coords).float()
    probe_dataset = TensorDataset(probe_tensor)
    probe_loader = DataLoader(probe_dataset, batch_size=65536, shuffle=False)

    with torch.no_grad():
        for (batch_coords,) in probe_loader:
            outputs = model(batch_coords.to(device))
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.append(preds)
    pred_map = torch.cat(all_preds).numpy().reshape(height, width)
    
    map_img=cv2.imread(map_path)[:,:,(2,1,0)]
    ax.imshow(map_img.mean(2), cmap='gray')

    ax.imshow(np.flipud(pred_map), extent=[0, width, height, 0], alpha=0.7, cmap='viridis')
    ax.axis('off')
    fig.tight_layout(pad=0)
    plt.show() 
    
# --- Main execution ---
if __name__ == "__main__":
    # --- Data Loading and Preprocessing ---
    nepal_color = np.array([255, 194, 75])
    outside_nepal_color = np.array([255, 255, 255])
    outside_nepal_threshold = 50
    nepal_threshold = 10000
    map_path = 'nepal-map.png'

    X_tensor, y_tensor, X_sample, y_sample, height, width = load_and_process_map(
        map_path,
        color_a=outside_nepal_color,
        color_b=nepal_color,
        threshold_a=outside_nepal_threshold,
        threshold_b=nepal_threshold,
        num_samples=10000
    )

    # --- Model Training ---
    # Hyperparameters
    num_epochs = 5
    random_seed = 25
    device = 'cpu'
    batch_size = 10000
    learning_rate = 0.005

    torch.manual_seed(random_seed)

    # Initialize and train the model
    layers = [36,32,32]
    model = BaarleNet(layers).to(device)
    model = train_model(model, X_tensor, y_tensor, X_sample, y_sample,
                        num_epochs=num_epochs, batch_size=batch_size,
                        learning_rate=learning_rate, device=device)

    viz_descision_boundary(model, height, width, map_path, device=device)


# referance - https://github.com/stephencwelch/manim_videos/blob/master/_2025/backprop_3/notebooks/Wide%20Training%20Example.ipynb
# Video - https://www.youtube.com/watch?v=qx7hirqgfuU
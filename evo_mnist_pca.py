import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from local_functions import *
from sklearn.decomposition import PCA

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)

# --------------------------------------------------------------
# Data Loading and Preprocessing
# --------------------------------------------------------------

# Load the data
mnist_data = pd.read_csv("data/mnist_test.csv").to_numpy()

# Subset data
n_per_digit = 200
mnist_data_list = []
for i in range(10):
    mnist_data_i = mnist_data[mnist_data[:,0] == i][:n_per_digit, :]
    mnist_data_list.append(mnist_data_i)
mnist_data = np.vstack(mnist_data_list)

# Format data
mnist_images = mnist_data[:, 1:] / 255.0  # Normalize
mnist_labels = mnist_data[:, 0]
mnist_images_tensor = torch.tensor(mnist_images, 
                                   dtype=torch.float32).to(device)

# weights
weights = np.ones(mnist_images.shape[0])
weights = weights / np.sum(weights)  # Normalize to sum to 1
weights = torch.tensor(weights, device=device, dtype=torch.float32)

# --------------------------------------------------------------
# Construct the input kernel
# --------------------------------------------------------------
  
# Make the input kernels
K_geom = compute_linear_kernel_torch(mnist_images_tensor, 
                                   weights=weights, device=device)
norm_geom = torch.sqrt(torch.trace(K_geom @ K_geom))
 
# Make the class labels kernel
Z = torch.zeros((mnist_images_tensor.shape[0], 10), device=device)
for i in range(10):
    Z[mnist_labels == i, i] = 1.0
K_class = compute_class_kernel_torch(mnist_images_tensor, Z,
                                     weights=weights, device=device)
norm_class = torch.sqrt(torch.trace(K_class @ K_class))

# --------------------------------------------------------------
# Compute the solution via RV descent
# --------------------------------------------------------------

# Mix the kernels 
alpha_vec = [0, 0.33, 0.66, 1]

# PCA solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(mnist_images), 
                     dtype=torch.float32)

Y_mix_list = []
RV_mix_list = []
for alpha in alpha_vec:
    
    K_mix = (1 - alpha) * K_geom / norm_geom + alpha * K_class / norm_class

    # The coordinates of mix kernel
    Y_mix, RV_mix = rv_ascent_torch(K_mix,
                                    compute_linear_kernel_torch,
                                    param=None,
                                    weights=weights,
                                    Y_0=Y_pca.to(device),
                                    device=device,
                                    conv_threshold=1e-6)
    Y_mix_list.append(Y_mix)
    RV_mix_list.append(RV_mix)

# --------------------------------------------------------------
# Plot the results in a row of subplots
# --------------------------------------------------------------

fig, axes = plt.subplots(1, len(alpha_vec), figsize=(20, 5))
for i, alpha in enumerate(alpha_vec):
    ax = axes[i]
    Y_mix = Y_mix_list[i]
    RV_mix = RV_mix_list[i]
    
    scatter = ax.scatter(Y_mix[:, 0].cpu(), Y_mix[:, 1].cpu(), 
                         c=mnist_labels, cmap='tab10', s=15)
    ax.set_title(f"alpha={alpha:.2f}, RV={RV_mix:.4f}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if i == 0:
        ax.legend(*scatter.legend_elements(), title="Digits")
plt.suptitle("MNIST RV with Input: Linear + Class,  Output: Linear")
plt.tight_layout()
plt.savefig("results/mnist/evo_pca.png", dpi=300)
plt.show()
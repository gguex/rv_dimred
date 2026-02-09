import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from local_functions import *
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

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
n_neighbors = 10
K_geom = compute_geodesic_kernel(mnist_images, param=n_neighbors, 
                                 weights=weights.to('cpu').numpy())
K_geom = torch.tensor(K_geom, dtype=torch.float32).to(device)
norm_geom = torch.sqrt(torch.trace(K_geom @ K_geom))
 
# Make the class labels kernel
Z = torch.zeros((mnist_images_tensor.shape[0], 10), device=device)
for i in range(10):
    Z[mnist_labels == i, i] = 1.0
K_class = compute_class_kernel_torch(mnist_images_tensor, Z,
                                     weights=weights, device=device)
norm_class = torch.sqrt(torch.trace(K_class @ K_class))

# Mix the kernels
alpha = 0.5
K_mix = (1 - alpha) * K_geom / norm_geom + alpha * K_class / norm_class

# --------------------------------------------------------------
# Compute the solution via RV descent
# --------------------------------------------------------------

# PCA solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(mnist_images), 
                     dtype=torch.float32)

# The coordinates of geometry kernel
Y_geom, RV_final_geom = rv_ascent_torch(K_geom, 
                                        compute_linear_kernel_torch, 
                                        param=None,
                                        weights=weights, 
                                        Y_0=Y_pca.to(device),
                                        device=device,
                                        conv_threshold=1e-8)

# The coordinates of mix kernel
Y_mix, RV_final_mix = rv_ascent_torch(K_mix,
                                      compute_linear_kernel_torch,
                                      param=None,
                                      weights=weights,
                                      Y_0=Y_pca.to(device),
                                      device=device,
                                      conv_threshold=1e-8)

# --------------------------------------------------------------
# Plot the results
# --------------------------------------------------------------

Y_geom_c = Y_geom.cpu().numpy()
RV_geom_c = RV_final_geom.cpu().numpy().item()
Y_mix_c = Y_mix.cpu().numpy()
RV_mix_c = RV_final_mix.cpu().numpy().item()

# Baseline PCA plot
iso = Isomap(n_components=2, n_neighbors=n_neighbors)
Y_iso = iso.fit_transform(mnist_images)
plt.figure(figsize=(8,6))
scatter = plt.scatter(Y_iso[:,0], Y_iso[:,1], c=mnist_labels, cmap='tab10', 
                      s=10)
plt.title(f"MNIST Isomap")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.savefig("results/mnist/comp_isomap.png", dpi=300)
plt.show() 

# Plot RV
plt.figure(figsize=(8,6))
scatter = plt.scatter(Y_geom_c[:,0], Y_geom_c[:,1], c=mnist_labels, cmap='tab10', 
                      s=10)
plt.title(f"MNIST RV with Input: Geodesic,  Output: Linear\n"
          f"RV: {RV_geom_c:.6f}")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.savefig("results/mnist/comp_isomap_rv.png", dpi=300)
plt.show()  

# Plot RV with class kernel
plt.figure(figsize=(8,6))
scatter = plt.scatter(Y_mix_c[:,0], Y_mix_c[:,1], c=mnist_labels, cmap='tab10', 
                      s=10)
plt.title(f"MNIST RV with Input: Geodesic + Class,  Output: Linear\n"
          f"RV: {RV_mix_c:.6f}")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, labels, title="Digit Label", loc="best")
plt.grid(True)
plt.savefig("results/mnist/comp_isomap_rv_class.png", dpi=300)
plt.show()  
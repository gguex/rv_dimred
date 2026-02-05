import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from local_functions import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Is cuda available?
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)

# Load the data
mnist_data = pd.read_csv("data/mnist_test.csv").to_numpy()

# Subset data
n_per_digit = 100
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

# Parameters for t-SNE
perplexity = 20
params, perp = binary_search_rbf_params(mnist_images, 
                                        target_perplexity=perplexity)
        
# Make the input kernels
K_in = compute_gaussP_kernel_torch(mnist_images_tensor, param=params, 
                                   weights=weights, device=device)

# PCA solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(mnist_images), 
                     dtype=torch.float32)

# The coordinates of outputs
Y_opt_torch, RV_final_torch = rv_descent_torch(K_in, compute_t_kernel_torch, 
                                               param=1,
                                               weights=weights, dim=2, lr=0.1, 
                                               Y_0=Y_pca.to(device),
                                               device=device,
                                               conv_threshold=1e-8)

# Plot the results
Y_opt = Y_opt_torch.cpu().numpy()
plt.figure(figsize=(8,6))
scatter = plt.scatter(Y_opt[:,0], Y_opt[:,1], c=mnist_labels, cmap='tab10', 
                      s=10)
plt.title(f"MNIST tSNE-RV with Perplexity={perplexity}")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.grid(True)
plt.savefig("results/mnist/mnist_tsne_rv.png", dpi=300)
plt.show()  

# Plot the TSNE with same perplexity for comparison
tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
Y_tsne = tsne.fit_transform(mnist_images)
plt.figure(figsize=(8,6))
scatter = plt.scatter(Y_tsne[:,0], Y_tsne[:,1], c=mnist_labels, cmap='tab10', 
                      s=10)
plt.title(f"MNIST tSNE with Perplexity={perplexity}")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.grid(True)
plt.savefig("results/mnist/mnist_tsne.png", dpi=300)
plt.show() 
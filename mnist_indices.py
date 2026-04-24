import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from local_functions import *
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE, trustworthiness
import umap

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
# Input and Output Kernels Construction
# --------------------------------------------------------------

# Parameters for t-SNE
n_neighbors = 12
perplexity = 30
gauss_params, _ = binary_search_rbf_params(mnist_images, 
                                           target_perplexity=perplexity)

# All input kernel functions to test
K_lin_in = compute_linear_kernel_torch(mnist_images_tensor, 
                                       param=None, 
                                       weights=weights, device=device)
K_geo_in_cpu = compute_geodesic_kernel(mnist_images, 
                                       param=n_neighbors,
                                       weights=weights.to('cpu').numpy())
K_geo_in = torch.tensor(K_geo_in_cpu, dtype=torch.float32).to(device)
K_lle_in = compute_lle_kernel_torch(mnist_images_tensor, 
                                    param=n_neighbors, 
                                    weights=weights, device=device)
K_gauss_in = compute_gaussP_kernel_torch(mnist_images_tensor, 
                                         param=gauss_params, 
                                         weights=weights, device=device)
K_topo_in = compute_fuzzy_topo_kernel_torch(mnist_images_tensor, 
                                            param=n_neighbors,
                                            weights=weights, device=device)

kernels_in = [K_lin_in, K_geo_in, K_lle_in, K_gauss_in, K_topo_in]

# All output kernel functions to test
kernel_out_functions = [compute_linear_kernel_torch,
                        compute_polynomial_kernel_torch,
                        compute_t_kernel_torch]
kernel_out_params = [None, None, 1]


# Combinations to test
combinations = [(0, 0), (1, 0), (2, 0), (4, 0), (3, 2), (4, 2)]
combination_names = ["Linear-Linear", "Geodesic-Linear", "LLE-Linear", "Fuzzy Topo.-Linear", 
                     "Adapt. Gaussian-Student", "Fuzzy Topo.-Student"]

# --------------------------------------------------------------
# Computations of the reference methods
# --------------------------------------------------------------

Y_pca = PCA(n_components=2).fit_transform(mnist_images)
Y_iso = Isomap(n_components=2, n_neighbors=n_neighbors).fit_transform(mnist_images)
Y_lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors).fit_transform(mnist_images)
Y_lap= SpectralEmbedding(n_components=2, n_neighbors=n_neighbors).fit_transform(mnist_images)
Y_tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(mnist_images)
Y_umap = umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(mnist_images)

Y_comp_list = [Y_pca, Y_iso, Y_lle, Y_lap, Y_tsne, Y_umap]

K_pca = compute_linear_kernel_torch(torch.tensor(Y_pca, dtype=torch.float32))
K_iso = compute_linear_kernel_torch(torch.tensor(Y_iso, dtype=torch.float32))
K_lle = compute_linear_kernel_torch(torch.tensor(Y_lle, dtype=torch.float32))
K_lap = compute_linear_kernel_torch(torch.tensor(Y_lap, dtype=torch.float32))
K_tsne = compute_linear_kernel_torch(torch.tensor(Y_tsne, dtype=torch.float32))
K_umap = compute_linear_kernel_torch(torch.tensor(Y_umap, dtype=torch.float32)) 

norm_pca = np.sqrt(np.trace(K_pca @ K_pca))
norm_iso = np.sqrt(np.trace(K_iso @ K_iso))
norm_lle = np.sqrt(np.trace(K_lle @ K_lle))
norm_lap = np.sqrt(np.trace(K_lap @ K_lap))
norm_tsne = np.sqrt(np.trace(K_tsne @ K_tsne))
norm_umap = np.sqrt(np.trace(K_umap @ K_umap))

k_comps = [(K_pca, norm_pca), (K_iso, norm_iso), (K_lle, norm_lle), (K_lap, norm_lap), 
           (K_tsne, norm_tsne), (K_umap, norm_umap)]

# --------------------------------------------------------------
# Computations of the combinations
# --------------------------------------------------------------

# Compute the MDS solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(
    mnist_images_tensor.cpu().numpy()))

# Compute the 16 possibles input-output combinations
RV_matrix = np.zeros((len(combinations), len(k_comps)))
thrust_matrix = np.zeros((len(combinations), len(k_comps)))
Y_opt_list = []
for i, combination in enumerate(combinations):
    
    in_index, out_index = combination
    
    K_in = kernels_in[in_index]
    output_kernel_function = kernel_out_functions[out_index]
    out_param = kernel_out_params[out_index]
    
    print(f"Compute {combination_names[i]}...\n")
            
    Y_opt_torch, RV_final_torch = rv_ascent_torch(K_in,
                                                  output_kernel_function, 
                                                  param=out_param, 
                                                  Y_0=Y_pca,
                                                  weights=weights,
                                                  conv_threshold=1e-7,
                                                  device=device)
    print(f"Final RV: {RV_final_torch.item()}\n")
    
    Y_opt = Y_opt_torch.detach().cpu()
    Y_opt_list.append(Y_opt.numpy())
    K_opt = compute_linear_kernel_torch(Y_opt)
    norm_opt = np.sqrt(np.trace(K_opt @ K_opt))
    
    RV_row = [np.trace(K_opt @ k_ref[0]) / (norm_opt * k_ref[1]) for k_ref in k_comps]
    thrust_row = [trustworthiness(Y_opt.numpy(), Y_comp, n_neighbors=n_neighbors) for Y_comp in Y_comp_list]
    RV_matrix[i, :] = RV_row
    thrust_matrix[i, :] = thrust_row
    
# --------------------------------------------------------------
# Plot of comparison results
# --------------------------------------------------------------

n_comp = len(combinations)

fig, axes = plt.subplots(n_comp, 2, figsize=(100/n_comp, 100/2))
for i in range(len(combinations)):
    ax = axes[i, 0]
    Y_opt = Y_opt_list[i]
    ax.scatter(Y_opt[:, 0], Y_opt[:, 1], c=mnist_labels, cmap='tab10', s=5)
    ax.set_title(combination_names[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[i, 1]
    
    ax = axes[i, 1]
    Y_comp = Y_comp_list[i]
    ax.scatter(Y_comp[:, 0], Y_comp[:, 1], c=mnist_labels, cmap='tab10', s=5)
    ax.set_title(f"{combination_names[i]} - Reference")
    ax.set_xticks([])
    ax.set_yticks([])   
plt.show()    

print(RV_matrix.round(5))
print(thrust_matrix.round(5))
    
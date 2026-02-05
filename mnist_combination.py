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
# Input and Output Kernels Construction
# --------------------------------------------------------------

# Parameters for t-SNE
perplexity = 40
gauss_params, _ = binary_search_rbf_params(mnist_images, 
                                     target_perplexity=perplexity)

# All input kernel functions to test
K_lin_in = compute_linear_kernel_torch(mnist_images_tensor, 
                                       param=None, 
                                       weights=weights, device=device)
K_geo_in_cpu = compute_geodesic_kernel(mnist_images, 
                                       param=10,
                                       weights=weights.to('cpu').numpy())
K_geo_in = torch.tensor(K_geo_in_cpu, dtype=torch.float32).to(device)
K_lle_in = compute_lle_kernel_torch(mnist_images_tensor, 
                                    param=15, 
                                    weights=weights, device=device)
K_gauss_in = compute_gaussP_kernel_torch(mnist_images_tensor, 
                                         param=gauss_params, 
                                         weights=weights, device=device)

kernels_in = [K_lin_in, K_geo_in, K_lle_in, K_gauss_in]

kernel_in_names = ['Linear', 'Geodesic', 'LLE', 'Gaussian']

# All output kernel functions to test
kernel_out_functions = [compute_linear_kernel_torch,
                        compute_rbf_kernel_torch,
                        compute_t_kernel_torch,
                        compute_cosine_kernel_torch]
kernel_out_params = [None, 1, 1, None]
kernel_out_names = ['Linear', 'RBF', 't', 'Cosine']

# --------------------------------------------------------------
# Computations of the combinations
# --------------------------------------------------------------

# Compute the MDS solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(
    mnist_images_tensor.cpu().numpy()))

# Compute the 16 possibles input-output combinations
RV_matrix_torch = np.zeros((4,4))
Y_opt_grid = []
for i in range(4):
    Y_opt_list = []
    for j in range(4):
        K_in = kernels_in[i]
        output_kernel_function = kernel_out_functions[j]
        out_param = kernel_out_params[j]
        
        print(f"Input: {kernel_in_names[i]}, "
              f"Output: {kernel_out_names[j]}\n")
              
        Y_opt_torch, RV_final_torch = rv_descent_torch(K_in,
                                                       output_kernel_function, 
                                                       param=out_param, 
                                                       Y_0=Y_pca,
                                                       weights=weights, 
                                                       conv_threshold=1e-6,
                                                       device=device)
        print(f"Final RV: {RV_final_torch.item()}\n")
        RV_matrix_torch[i,j] = RV_final_torch.item()
        Y_opt_list.append(Y_opt_torch.detach().cpu().numpy())
    Y_opt_grid.append(Y_opt_list)

# --------------------------------------------------------------
# Plotting the Results
# --------------------------------------------------------------

# Plot the results in a grid with the label colors
c_color = mnist_labels
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        Y_opt = Y_opt_grid[i][j]
        ax.scatter(Y_opt[:, 0], Y_opt[:, 1], c=c_color, cmap='tab10', s=5)
        ax.set_title(f"Input: {kernel_in_names[i]}, "
                     f"Output: {kernel_out_names[j]}\n"
                     f"RV: {RV_matrix_torch[i,j]:.6f}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.savefig("results/mnist/mnist_combinations2.png", dpi=300)
                                                    
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

# Load the data
mnist_data = pd.read_csv("data/mnist_test.csv").to_numpy()

# Subset data
mnist_data_list = []
for i in range(10):
    mnist_data_i = mnist_data[mnist_data[:,0] == i][:200, :]
    mnist_data_list.append(mnist_data_i)
mnist_data = np.vstack(mnist_data_list)

# Format data
mnist_images = mnist_data[:, 1:] / 255.0  # Normalize
mnist_labels = mnist_data[:, 0]
mnist_images_tensor = torch.tensor(mnist_images, 
                                   dtype=torch.float32).to(device)
weights = torch.ones(mnist_images_tensor.shape[0], device=device) / mnist_images_tensor.shape[0]

# Define the input kernels parameters to test
kernels_in_params = [None, 3, 0.5, 3]

# Make all the input kernels
K_lin_in = compute_linear_kernel_torch(mnist_images_tensor, param=kernels_in_params[0], 
                                       weights=weights, device=device)
K_poly_in = compute_polynomial_kernel_torch(mnist_images_tensor, param=kernels_in_params[1], 
                                            weights=weights, device=device)
K_rbf_in = compute_rbf_kernel_torch(mnist_images_tensor, param=kernels_in_params[2], 
                                    weights=weights, device=device)
K_t_in = compute_t_kernel_torch(mnist_images_tensor, param=kernels_in_params[3], 
                                weights=weights, device=device)

# The normalization factors
N_lin = torch.sqrt(torch.trace(K_lin_in @ K_lin_in))
N_poly = torch.sqrt(torch.trace(K_poly_in @ K_poly_in))
N_rbf = torch.sqrt(torch.trace(K_rbf_in @ K_rbf_in))
N_t = torch.sqrt(torch.trace(K_t_in @ K_t_in))

# the 4 outputs for all the input kernels
kernels_in = [K_lin_in/N_lin, K_poly_in/N_poly, K_rbf_in/N_rbf, K_t_in/N_t]
kernels_names = ['Linear', 'Polynomial', 'RBF', 't']
kernels_out_functions = [compute_linear_kernel_torch, 
                         compute_polynomial_kernel_torch, 
                         compute_rbf_kernel_torch, 
                         compute_t_kernel_torch]
# Define the output kernels parameters to test
kernels_out_params = [None, 1, 1, 1]

# Compute the MDS solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(mnist_images_tensor.cpu().numpy()))

# Compute the 16 possibles input-output combinations
RV_matrix_torch = np.zeros((4,4))
Y_opt_grid = []
for i in range(4):
    Y_opt_list = []
    for j in range(4):
        K_in = kernels_in[i]
        output_kernel_function = kernels_out_functions[j]
        param = kernels_out_params[j]
        
        Y_opt_torch, RV_final_torch = rv_descent_torch(K_in, output_kernel_function, 
                                                       param=param, Y_0=Y_pca,
                                                       weights=weights, dim=2, lr=0.1, device=device)
        RV_matrix_torch[i,j] = RV_final_torch.item()
        Y_opt_list.append(Y_opt_torch.cpu().numpy())
        print(f"Input: {kernels_names[i]}, Output: {kernels_names[j]}, RV: {RV_final_torch.item():.6f}")
    Y_opt_grid.append(Y_opt_list)


# Plot the results in a grid with the label colors
c_color = mnist_labels
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        Y_opt = Y_opt_grid[i][j]
        ax.scatter(Y_opt[:, 0], Y_opt[:, 1], c=c_color, cmap='tab10', s=5)
        ax.set_title(f"Input: {kernels_names[i]}, Output: {kernels_names[j]}\nRV: {RV_matrix_torch[i,j]:.6f}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.savefig("results/mnist/rv_mnist_std.png", dpi=300)
                                                    
import numpy as np
import matplotlib.pyplot as plt
import torch
from local_functions import *

# Test the torch implementation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Make the grid
my_grid = torch.tensor(create_lattice(20)).to(device).to(torch.float32)
weights = torch.ones(my_grid.shape[0], device=device) / my_grid.shape[0]

# Define the input and output kernels parameters to test
kernels_params = [None, 1, 0.5, 2]

# Make all the input kernels
K_lin_in = compute_linear_kernel_torch(my_grid, param=kernels_params[0], 
                                       weights=weights, device=device)
K_poly_in = compute_polynomial_kernel_torch(my_grid, param=kernels_params[1], 
                                            weights=weights, device=device)
K_rbf_in = compute_rbf_kernel_torch(my_grid, param=kernels_params[2], 
                                    weights=weights, device=device)
K_t_in = compute_t_kernel_torch(my_grid, param=kernels_params[3], 
                                weights=weights, device=device)

# the 4 outputs for all the input kernels
kernels_in = [K_lin_in, K_poly_in, K_rbf_in, K_t_in]
kernels_names = ['Linear', 'Polynomial', 'RBF', 't']
kernels_out_functions = [compute_linear_kernel_torch, 
                         compute_polynomial_kernel_torch, 
                         compute_rbf_kernel_torch, 
                         compute_t_kernel_torch]

# Compute the 16 possibles input-output combinations
RV_matrix_torch = np.zeros((4,4))
Y_opt_grid = []
for i in range(4):
    Y_opt_list = []
    for j in range(4):
        K_in = kernels_in[i]
        output_kernel_function = kernels_out_functions[j]
        param = kernels_params[j]
        
        Y_opt_torch, RV_final_torch = rv_descent_torch(K_in, output_kernel_function, 
                                                       param=param, Y_0=None,
                                                       weights=weights, dim=2, lr=0.1, device=device)
        RV_matrix_torch[i,j] = RV_final_torch.item()
        Y_opt_list.append(Y_opt_torch.cpu().numpy())
        print(f"Input: {kernels_names[i]}, Output: {kernels_names[j]}, RV: {RV_final_torch.item():.6f}")
    Y_opt_grid.append(Y_opt_list)


# Plot the results in a grid with the same rainbow color
c_color=np.linalg.norm(my_grid.cpu(), axis=1)
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        Y_opt = Y_opt_grid[i][j]
        ax.scatter(Y_opt[:, 0], Y_opt[:, 1], c=c_color, cmap='rainbow', s=5)
        ax.set_title(f"Input: {kernels_names[i]}, Output: {kernels_names[j]}\nRV: {RV_matrix_torch[i,j]:.6f}")
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.savefig("results/grid/rv_grid.png", dpi=300)
                                                    
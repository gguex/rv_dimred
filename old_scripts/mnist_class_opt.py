import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from local_functions import *
from sklearn.decomposition import PCA

# Is cuda available?
device = "cuda" if torch.cuda.is_available() else "cpu"
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
mnist_images_tensor = torch.tensor(mnist_images).to(device).to(torch.float32)
weights = torch.ones(mnist_images_tensor.shape[0], device=device) / mnist_images_tensor.shape[0]

# Make the class kernel
Z = torch.zeros((mnist_images_tensor.shape[0], 10), device=device)
for i in range(10):
    Z[mnist_labels == i, i] = 1.0
P_Z = Z @ torch.linalg.pinv(Z.T @ torch.diag(weights) @ Z) @ Z.T @ torch.diag(weights)
K_Z = P_Z @ mnist_images_tensor @ mnist_images_tensor.T @ P_Z.T

# The normalization factor
N_Z = torch.sqrt(torch.trace(K_Z @ K_Z))

# Make a mixted kernel as input
alpha = 1

# The value of the parameter for the output kernel
params_in = [0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]

# Compute the MDS solution for reference
Y_pca = torch.tensor(PCA(n_components=2).fit_transform(mnist_images_tensor.cpu().numpy()))

K_lin = compute_linear_kernel_torch(mnist_images_tensor, param=None, 
                                   weights=weights, device=device)

# Compute the 16 possibles input-output combinations
RV_list = []
RV_init = []
Y_opt_list = []
for param_in in params_in:
    print(f"Computing for input kernel param={param_in}")
    K_rbf_in = compute_rbf_kernel_torch(mnist_images_tensor, param=param_in, 
                                        weights=weights, device=device)
    N_rbf = torch.sqrt(torch.trace(K_rbf_in @ K_rbf_in))
    K_rbf_mix_in = alpha * K_rbf_in / N_rbf + (1- alpha) * K_Z / N_Z
    Y_opt_torch, RV_final_torch = rv_descent_torch(K_rbf_mix_in, 
                                                   compute_t_kernel_torch, 
                                                   param=1, 
                                                   Y_0=Y_pca,
                                                   weights=weights, dim=2, 
                                                   conv_threshold = 1e-5,
                                                   lr=0.1, device=device)
    New_K = compute_t_kernel_torch(Y_opt_torch.detach(), param=1, 
                                   weights=weights, device=device)
    RV_from_init = compute_rv(K_lin, New_K)
    RV_list.append(RV_final_torch.item())
    RV_init.append(RV_from_init.item())
    Y_opt_list.append(Y_opt_torch.cpu().numpy())
    
# Plot the RV values as a function of the output kernel parameter
plt.figure(figsize=(8,6))
plt.plot(params_in, RV_list, marker='o')
plt.xlabel("Output RBF kernel parameter")
plt.ylabel("Final RV value")
plt.title("RV values vs RBF input kernel parameter")
plt.grid()
plt.savefig("results/mnist/rv_mnist_cls_param.png", dpi=300)

# Plot the RV values as a function of the output kernel parameter
plt.figure(figsize=(8,6))
plt.plot(params_in, RV_init, marker='o')
plt.xlabel("Output RBF kernel parameter")
plt.ylabel("Final RV value")
plt.title("RV values vs RBF input kernel parameter")
plt.grid()
plt.savefig("results/mnist/rv_init_mnist_cls_param.png", dpi=300)

# Plot the embeddings for the max RV value
max_index = np.argmax(RV_list)
Y_opt = Y_opt_list[max_index]
c_color = mnist_labels
plt.figure(figsize=(8,6))
plt.scatter(Y_opt[:, 0], Y_opt[:, 1], c=c_color, cmap='tab10', s=20)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title(f"Optimal embedding for RBF input kernel param={params_in[max_index]}, RV={RV_list[max_index]:.6f}")
plt.colorbar(ticks=range(10), label='Digit Label')
plt.savefig("results/mnist/rv_mnist_cls_embedding.png", dpi=300)
                                                    
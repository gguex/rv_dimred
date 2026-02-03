import numpy as np
import matplotlib.pyplot as plt
import torch
from local_functions import *

# Test the torch implementation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

my_grid = torch.tensor(create_lattice(20)).to(device)

weights = torch.ones(my_grid.shape[0], device=device) / my_grid.shape[0]
my_grid_torch = torch.tensor(my_grid, dtype=torch.float32, device=device)
K_obj_torch = compute_rbf_kernel_torch(my_grid_torch, weights=weights, device=device)

Y_opt_torch, RV_final_torch = rv_descent_torch(K_obj_torch, compute_rbf_kernel_torch, 
                                               weights=weights, dim=2, lr=0.01, device=device)

Y_opt = Y_opt_torch.cpu().numpy()

# Plot the optimized points
plt.scatter(Y_opt[:, 0], Y_opt[:, 1], c=np.linalg.norm(Y_opt, axis=1), 
            cmap='rainbow')
plt.title('Optimized 2D Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()    
                                                    



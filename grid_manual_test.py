import numpy as np
import matplotlib.pyplot as plt
from local_functions import *

my_grid = create_lattice(20)

# Display the grid points with a raindbow colormap
plt.scatter(my_grid[:, 0], my_grid[:, 1], c=np.linalg.norm(my_grid, axis=1), 
            cmap='rainbow')
plt.title('2D Lattice')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()

weights = np.ones(my_grid.shape[0]) / my_grid.shape[0]
K_obj = compute_rbf_kernel(my_grid, weights=weights)
Y_opt, RV_final = rv_descent(K_obj, weights, dim=2, lr=0.01)

# Plot the optimized points
plt.scatter(Y_opt[:, 0], Y_opt[:, 1], c=np.linalg.norm(Y_opt, axis=1), 
            cmap='rainbow')
plt.title('Optimized 2D Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()  
                                                    



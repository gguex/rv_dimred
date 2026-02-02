import numpy as np
import matplotlib.pyplot as plt

# Define a lattice of points in 2D
def create_lattice(n_side, mesh_size=1):
    x = np.tile(np.linspace(1, mesh_size*n_side, n_side), n_side)
    y = np.repeat(np.linspace(1, mesh_size*n_side, n_side), n_side)
    
    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    
my_grid = create_lattice(20)


# Display the grid points with a raindbow colormap
plt.scatter(my_grid[:, 0], my_grid[:, 1], c=np.linalg.norm(my_grid, axis=1), 
            cmap='rainbow')
plt.title('2D Lattice')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()

# Compute the kernel from coordinates
def compute_kernel(coords, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    K_mat = Q_mat @ coords @ coords.T @ Q_mat.T
    return K_mat

# Make the algorithm of gradient descent
def rv_descent(K_obj, weights, dim=2, lr=0.1, 
               conv_threshold = 1e-8, 
               n_iter_max=10000):
    
    n = K_obj.shape[0]
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    
    Norm_obj = np.sqrt(np.trace(K_obj @ K_obj))
    
    Y_0 = np.random.normal(size=(n, dim))
    Y = Y_0.copy()
    RV_prev = 1000
    for i in range(n_iter_max):
        K_Y = Q_mat @ Y @ Y.T @ Q_mat.T
        Norm_Y = np.sqrt(np.trace(K_Y @ K_Y))
        Scal_Obj_Y = np.trace(K_obj @ K_Y)
        
        RV = Scal_Obj_Y / (Norm_obj * Norm_Y)
        M = 1/(Norm_obj * Norm_Y) * (K_obj - Scal_Obj_Y / Norm_Y**2 * K_Y)
        grad = 2 * M @ Y
        
        Y += lr * grad
        
        if i % 100 == 0:
            print(f"Iteration {i+1}: RV = {RV}")
        
        if np.abs(RV - RV_prev) < conv_threshold:
            print("Convergence reached.")
            break
        RV_prev = RV
    
    return Y, RV


K_obj = compute_kernel(my_grid)
weights = np.ones(my_grid.shape[0]) / my_grid.shape[0]
Y_opt, RV_final = rv_descent(K_obj, weights, dim=2, lr=0.01)


# Plot the optimized points
plt.scatter(Y_opt[:, 0], Y_opt[:, 1], c=np.linalg.norm(Y_opt, axis=1), 
            cmap='rainbow')
plt.title('Optimized 2D Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()  
                                                    
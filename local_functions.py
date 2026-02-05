import numpy as np
import torch
import scipy.sparse.csgraph as csgraph
from sklearn.neighbors import kneighbors_graph

#------------------------------------
#-------------- CPU VERSIONS
#------------------------------------

# Compute the kernel from coordinates
def compute_linear_kernel(coords, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    K_mat = Q_mat @ coords @ coords.T @ Q_mat.T
    return K_mat

# Compute the polynomial kernel from coordinates
def compute_polynomial_kernel(coords, gamma=2, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    
    G_mat = (gamma * coords @ coords.T)**3
    K_mat = Q_mat @ G_mat @ Q_mat.T
    return K_mat

# Compute the Gaussian kernel from coordinates
def compute_rbf_kernel(coords, gamma=1, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    
    pairwise_dists = np.sum(coords**2, axis=1).reshape(-1, 1) + \
                     np.sum(coords**2, axis=1) - 2 * coords @ coords.T
    G_gauss = np.exp(-gamma * pairwise_dists)
    K_mat = Q_mat @ G_gauss @ Q_mat.T
    return K_mat

# Compute t-kernel from coordinates
def compute_t_kernel(coords, df=3, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    
    pairwise_dists = np.sum(coords**2, axis=1).reshape(-1, 1) + \
                     np.sum(coords**2, axis=1) - 2 * coords @ coords.T
    G_t = (1 + pairwise_dists / df) ** (-(df + 1) / 2)
    K_mat = Q_mat @ G_t @ Q_mat.T
    return K_mat

# Make the algorithm of gradient descent
def rv_descent(K_obj, weights, dim=2, lr=0.1, 
               conv_threshold = 1e-5, 
               n_iter_max=50000):
    
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
        
        if i % 500 == 0:
            print(f"Iteration {i+1}: RV = {RV}")
        
        if np.abs(RV - RV_prev) < conv_threshold:
            print("Convergence reached.")
            break
        RV_prev = RV
    
    return Y, RV

#------------------------------------
#-------------- TORCH VERSIONS
#------------------------------------

def compute_linear_kernel_torch(coords, param=None, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    K_mat = Q_mat @ coords @ coords.T @ Q_mat.T
    return K_mat

def compute_polynomial_kernel_torch(coords, param=None, degree=3, coef0=1, weights=None, device='cpu'):
    n = coords.shape[0]
    p = coords.shape[1]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    if param is None:
        param = 1 / p
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    G_mat = (param * coords @ coords.T + coef0)**degree
    K_mat = Q_mat @ G_mat @ Q_mat.T
    return K_mat

def compute_rbf_kernel_torch(coords, param=1, weights=None, device='cpu'):
    n = coords.shape[0]
    param = torch.tensor(param, device=device, dtype=torch.float32)
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
    if param.ndim == 0:
        G_gauss = torch.exp(-param * pairwise_dists)
    else:
        G_gauss = torch.exp(-param[:, None] * pairwise_dists)
        # Symmetrize the kernel
        G_gauss = (G_gauss + G_gauss.T) / 2
    K_mat = Q_mat @ G_gauss @ Q_mat.T
    return K_mat    

# Create the isomap kernel (CPU only, i.e. input only)
def compute_geodesic_kernel(coords, param=10, weights=None):
    n = coords.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    H_mat = np.eye(n) - np.outer(np.ones(n), weights)
    Q_mat = np.diag(np.sqrt(weights)) @ H_mat
    
    adjacency = kneighbors_graph(coords,
                                 n_neighbors=param,
                                 mode='distance', include_self=False)
    sp_dists = csgraph.shortest_path(adjacency, method='auto', 
                                     directed=False)
    pairwise_dists = ((sp_dists + sp_dists.T) / 2)**2
    
    K_mat = -0.5 * Q_mat @ pairwise_dists @ Q_mat.T
    return K_mat 

def compute_gaussP_kernel_torch(coords, param=1, power=0.5, weights=None, 
                                device='cpu'):
    
    n = coords.shape[0]
    param = torch.tensor(param, device=device, dtype=torch.float32)
    
    if weights is None:
        weights = torch.ones(n, device=device) / n
        
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
                     
    if param.ndim == 0:
        G_gauss = torch.exp(-param * pairwise_dists)
    else:
        G_gauss = torch.exp(-param[:, None] * pairwise_dists)
    
    G_gauss.fill_diagonal_(0)
    sums_gauss = torch.sum(G_gauss, axis=1)
    G_P = G_gauss / (sums_gauss[:, np.newaxis] + 1e-15)
    G_P = (G_P + G_P.T) / 2
    G_P = G_P**power
    
    K_mat = Q_mat @ G_P @ Q_mat.T
    
    return K_mat    

def compute_t_kernel_torch(coords, param=1, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
    G_t = (1 + pairwise_dists / param) ** (-1)
    G_t.fill_diagonal_(0)

    K_mat = Q_mat @ G_t @ Q_mat.T
    return K_mat

def compute_tP_kernel_torch(coords, param=1, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
    G_t = (1 + pairwise_dists / param) ** (-(param + 1) / 2)
    
    G_t[np.arange(G_t.shape[0]), np.arange(G_t.shape[0])] = 0
    sums_t = torch.sum(G_t, axis=1)
    G_P = G_t / (sums_t[:, np.newaxis] + 1e-15)
    G_P = (G_P + G_P.T) / 2
    
    K_mat = Q_mat @ G_P @ Q_mat.T
    return K_mat

def compute_rv(K_in, K_out):
    Norm_in = torch.sqrt(torch.trace(K_in @ K_in))
    Norm_out = torch.sqrt(torch.trace(K_out @ K_out))
    Scal_Obj_Y = torch.trace(K_in @ K_out)
    
    RV = Scal_Obj_Y / (Norm_in * Norm_out)
    return RV

def rv_descent_torch(K_in, output_kernel_function, param, Y_0=None, weights=None, dim=2, lr=0.1, 
                     conv_threshold = 1e-5, 
                     n_iter_max=50000, device='cpu'):
    
    n = K_in.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
        
    if Y_0 is not None:
        Y = Y_0.to(device)
        Y.requires_grad = True
    else:
        Y = torch.normal(0, 1, size=(n, dim), device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([Y], lr=lr, maximize=True)
    
    RV_old = 1000
    for i in range(n_iter_max):
        optimizer.zero_grad()
        K_out = output_kernel_function(Y, param=param, weights=weights, device=device)
        RV = compute_rv(K_in, K_out)
        RV.backward()
        optimizer.step()
        
        if i % 500 == 0:
            print(f"Iteration {i+1}: RV = {RV.item()}")
        
        if torch.abs(RV - RV_old) < conv_threshold:
            print("Convergence reached.")
            break
        
        RV_old = RV.clone().detach()
    return Y.detach(), RV.detach()


#------------------------------------
#-------------- UTILITY FUNCTIONS
#------------------------------------

# Define a lattice of points in 2D
def create_lattice(n_side, mesh_size=1):
    x = np.tile(np.linspace(1, mesh_size*n_side, n_side), n_side)
    y = np.repeat(np.linspace(1, mesh_size*n_side, n_side), n_side)
    
    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

# Binary search for optimal parameter with gaussian kernel given 
# a target perplexity
def binary_search_rbf_params(coord, target_perplexity, tolerance=1e-5, 
                             max_iter=1000):
    
    # Get the number of points
    n = coord.shape[0]
    
    # Compute the squared distances matrix
    sq_dist = np.sum(coord**2, axis=1).reshape(-1, 1) + \
               np.sum(coord**2, axis=1) - 2 * coord @ coord.T
    
    # Target entropy
    target_entropy = np.log2(target_perplexity)
    
    # Initialize bounds
    betas_min = np.zeros(n)
    betas_max = np.ones(n) * np.inf
    betas = np.ones(n)
    
    for i in range(max_iter):
        
        # Compute the Gaussian kernel and entropy on every row 
        probs = np.exp(-betas[:, np.newaxis] * sq_dist)
        np.fill_diagonal(probs, 0)
        sum_probs = np.sum(probs, axis=1) - np.diag(probs)
        P = probs / (sum_probs[:, np.newaxis] + 1e-15)
        entropies = -np.sum(P * np.log2(P + 1e-10), axis=1)
        
        # Check the difference between computed and target entropy
        entropies_diff = entropies - target_entropy
        
        # Check convergence
        if np.all(np.abs(entropies_diff) < tolerance):
            break
        
        # Update beta for each point
        for j in range(n):
            if entropies_diff[j] > 0:
                betas_min[j] = betas[j]
                if betas_max[j] == np.inf:
                    betas[j] *= 2
                else:
                    betas[j] = (betas[j] + betas_max[j]) / 2
            else:
                betas_max[j] = betas[j]
                if betas_min[j] == 0:
                    betas[j] /= 2
                else:
                    betas[j] = (betas[j] + betas_min[j]) / 2
                
    return betas, 2**(entropies)
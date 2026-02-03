import numpy as np
import torch

# Define a lattice of points in 2D
def create_lattice(n_side, mesh_size=1):
    x = np.tile(np.linspace(1, mesh_size*n_side, n_side), n_side)
    y = np.repeat(np.linspace(1, mesh_size*n_side, n_side), n_side)
    
    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

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
    
    G = (gamma * coords @ coords.T)**3
    K_mat = Q_mat @ G @ Q_mat.T
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
    K_gauss = np.exp(-gamma * pairwise_dists)
    K_mat = Q_mat @ K_gauss @ Q_mat.T
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
    K_t = (1 + pairwise_dists / df) ** (-(df + 1) / 2)
    K_mat = Q_mat @ K_t @ Q_mat.T
    return K_mat

# Make the algorithm of gradient descent
def rv_descent(K_obj, weights, dim=2, lr=0.1, 
               conv_threshold = 1e-8, 
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

#-------------- TORCH VERSION

def compute_linear_kernel_torch(coords, param=None, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    K_mat = Q_mat @ coords @ coords.T @ Q_mat.T
    return K_mat

def compute_polynomial_kernel_torch(coords, param=2, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    G = (param * coords @ coords.T)**3
    K_mat = Q_mat @ G @ Q_mat.T
    return K_mat
    

def compute_t_kernel_torch(coords, param=3, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
    K_t = (1 + pairwise_dists / param) ** (-(param + 1) / 2)
    K_mat = Q_mat @ K_t @ Q_mat.T
    return K_mat

def compute_rbf_kernel_torch(coords, param=1, weights=None, device='cpu'):
    n = coords.shape[0]
    if weights is None:
        weights = torch.ones(n, device=device) / n
    H_mat = torch.eye(n, device=device) - torch.outer(torch.ones(n, device=device), weights)
    Q_mat = torch.diag(torch.sqrt(weights)) @ H_mat
    
    pairwise_dists = torch.sum(coords**2, axis=1).reshape(-1, 1) + \
                     torch.sum(coords**2, axis=1) - 2 * coords @ coords.T
    K_gauss = torch.exp(-param * pairwise_dists)
    K_mat = Q_mat @ K_gauss @ Q_mat.T
    return K_mat    

def compute_rv(K_in, K_out):
    Norm_in = torch.sqrt(torch.trace(K_in @ K_in))
    Norm_out = torch.sqrt(torch.trace(K_out @ K_out))
    Scal_Obj_Y = torch.trace(K_in @ K_out)
    
    RV = Scal_Obj_Y / (Norm_in * Norm_out)
    return RV

def rv_descent_torch(K_in, output_kernel_function, param, Y_0=None, weights=None, dim=2, lr=0.1, 
                     conv_threshold = 1e-8, 
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

5 Case Studies
Overview of the section

5.1 Tested methods

5.1.1 Spectral methods:
- PCA
- Kernel PCA (RBF)
- Isomap
- LLE
- Diffusion Maps
5.1.2 Approximations:
- t-SNE
- UMAP
5.1.3 Hybrid methods:
- Golbal-Local, Light-Heavy Tails Interpolations :
    Input: $\alpha K_{linear} + (1 - \alpha) K_{adaptative gaussian}$; Output: $K^\nu_{student}$
- Unsupervised-Supervised Interpolation:
    Input: $\alpha K_{adaptative gaussian} + (1 - \alpha) K_{class}$

5.2 Experimental design
Overview of the experimental design

5.2.1 Datasets
- Single-Cell
- MNIST 
- Swiss-Roll

5.2.3 Similarity and quality indices
- Procrustes (similarity)
- kNN overlap (similarity)
- Trustworthiness (quality)
- Q_nx (quality)

5.3 Results 

5.3.1 Spectral methods
- Table of indices of the 3 datasets
- 3 results for the reference methods and the kernels combinations (6 scatterplots - methods and dataset to select).
5.3.2 Approximations
- Table of indices of the 3 datasets
- Results for the reference methods and the kernels combinations (4 scatterplots - dataset to select).
5.3.3 Hybrid methods
- Grid of interpolated results regarding $\alpha$ and $\nu$ (9 scatterplots - dataset to select).
- 3 scatterplots for interpolation regarding $\alpha$ (dataset to select).


## NEW

5 Case Studies
Overview of the section

5.1 Tested methods
(Every time, describe the input and output kernel, and find a short mathematical name like K_something)
5.1.1 Spectral methods:
(State why using the output kernels )
- PCA
- Kernel PCA (RBF)
- Isomap
- LLE
- Diffusion Maps
- Laplacian 
5.1.2 Approximations:
- t-SNE
- UMAP
5.1.3 Hybrid methods:
- Golbal-Local, Light-Heavy Tails Interpolations :
    Input: $\alpha K_{linear} + (1 - \alpha) K_{adaptative gaussian}$; Output: $K^\nu_{student}$
- Unsupervised-Supervised Interpolation:
    Input: $\alpha K_{adaptative gaussian} + (1 - \alpha) K_{class}$

5.2 Experimental design
Overview of the experimental design

5.2.1 Datasets
- Single-Cell
- MNIST 
- Swiss-Roll

5.2.3 Similarity and quality indices
- Procrustes (similarity)
- kNN overlap (similarity)
- Trustworthiness (quality)
- ARI (quality – if there are labels)

5.3 Results 

5.3.1 Spectral methods
- Table of the 4 indices, for all methods for the 3 datasets (not ARI for swissroll).
- 6 graphics : 1 row for the kernel method 1 row for the references : MNIST-PCA, ISOMAP-SINGLECELL, DIFFUSION-SWISSROLL)
- Discussions
5.3.2 Approximations
- Table of the 4 indices, for all methods for the 3 datasets (not ARI for swissroll).
- Results for the reference methods and the kernels combinations (4 scatterplots - dataset to select).
5.3.3 Hybrid methods
- Grid of interpolated results regarding $\alpha$ and $\nu$ (9 scatterplots - dataset to select).
- 3 scatterplots for interpolation regarding $\alpha$ (dataset to select).




# Nouvelles itérations 

## Itération 1

Nous allons maintenant refaire les résultats 5.3.1 et 5.3.2, de la manière suivante :

NE TOUCHE PAS AUX PARTIES QUI CONCERNENT LES RESULTATS 5.3.3.

1) Laplacian : modifie le kernel pour qu'il soit en accord avec la méthode SpectralEmbedding.
2) LLE : aligne reg sur sklearn.
3) DIFFUSION : augmente le t=10 comme prévu. 
4) Adapte le code pour avoir les résultats suivants :
   - Ne réduit PAS la dimensionnalité des données avant de faire les tests (i.e. MNIST avec toutes les dimensions).
   - Lorsque tu produits de comparaisons "méthode kernel - baseline", fais **2 graphiques**, dans **2 fichiers séparés**, avec nom adéquats (je vais les coller côte à côte dans le LateX).
   - Abandonne les calculs du Qnx, les courbes vont prendre trop de place pour être incluses dans l'article. Les indices restants sont: Procrustes, kNN overlap, Trustworthiness et ARI (s'il y a a des labels).
   - Pour t-SNE et UMAP (baseline) ne fait QUE les versions avec initialisation PCA.
5) NETTOYE TOUS LES FICHIERS DES CODES, RESULTATS PRECEDANTS ET TESTS/DIAGNOSTIQUES: IL NE DOIT RESTER QUE LES RESULTATS ACTUELS ET LES CODES LES AYANT PRODUIT.

## Itération 2


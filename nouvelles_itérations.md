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

NOUVEAU FORMAT DE SORTIE (RESULTATS), A APPLIQUER POUR LE MOMENT SUR LA PARTIE 5.3.1 : 
- SAUVE LES COORDONNEES RESULTANTES (KERNELS ET REFERENCES) dans results/coordinates/spectral/.
- Fais 2 scripts complémentaires qui, à partir des ces coordonnées: (1) calcule les différents indices (2) construit les graphiques. 

En ayant cela en tête, fais les opérations suivantes
1) Construit ce nouveau kernel "projecteur" de sortie, n'oublie pas de le rendre pondéré et centré, avec la matrice Q = diag(sqrt(f)) (I - 1 f^T).
2) Refais toutes les méthodes spectrales du point 5.3.1 avec ce kernel de sortie ET refais aussi touts les résultats précédants, avec kernel de sortie linéaire.
3) Produit les résultats indépendamment, et fais des comparaisons pour voir quel kernel de sortie fonctionne le mieux.




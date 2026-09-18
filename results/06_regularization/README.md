# Essais de régularisation dans l'espace des noyaux

Ce dossier regroupe les résultats des vérifications et des essais de la
régularisation par contrôle de la composante neutre.

## Résultats disponibles

Le sous-dossier `math_checks/` contient les résultats du script
[verify_kernel_regularization.py](../../scripts/exploratory/verify_kernel_regularization.py) :

- [checks.json](math_checks/checks.json) : erreurs numériques des identités,
  gradients, plafond spectral et contre-exemple de séparation de deux groupes ;
- [trace_regularization.pdf](math_checks/trace_regularization.pdf) et
  [trace_regularization.png](math_checks/trace_regularization.png) : illustration
  de la trace et de la pénalité le long de dilatations.

Pour les régénérer depuis la racine du dépôt :

```sh
.venv/bin/python scripts/exploratory/verify_kernel_regularization.py
```

Ces résultats sont des vérifications mathématiques, pas une évaluation des
performances d'un algorithme de réduction de dimension. Le script conserve
aussi des contrôles de variantes exploratoires qui ne sont pas retenues pour
la révision.

## Intégration et exemple contrôlé

L'expérience prévue a été réalisée le 18 septembre 2026 : trois intensités
de pénalisation, dont zéro, sur les mêmes trois initialisations. Le
[compte rendu](controlled_example/README.md) donne le protocole, les résultats
et les limites. Les neuf essais sont conservés, sans sélection du meilleur.

La pénalité rapproche l'inertie du noyau de la cible, au prix d'une diminution
du RV. La pénalité forte produit aussi des points éloignés et une sensibilité
à l'initialisation. Cet exemple illustre un contrôle d'inertie ; il ne justifie
pas une amélioration générale de la visualisation ni une garantie de convergence.

Les cinq tests d'intégration du solveur passent : projection de Frobenius et
forces, trajectoires inchangées à pénalité nulle, optimisation de l'objectif
pénalisé, cohérence du score final et du suivi, validation des paramètres.
[Résultat des tests](solver_checks.json).

```sh
.venv/bin/python scripts/exploratory/verify_regularized_solver.py
.venv/bin/python scripts/experiments/06_regularization/regularization_run.py
```

## Comparaison MNIST retenue dans l'article

Le dossier [mnist_hollow_regularization](mnist_hollow_regularization/) reprend
exactement les 2 000 observations, les coordonnées full-RV, hollow-RV et t-SNE,
et les trajectoires de l'ancienne expérience `04_tether`. À partir de la solution
full-RV historique, quatre essais appariés de 500 pas comparent hollow RV, full
RV à `eta=100` et `eta=1000`, et hollow RV régularisé à `eta=100000`.

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_hollow_regularization.py
```

## Chemin MNIST préliminaire

Le dossier [mnist_path](mnist_path/README.md) conserve la visualisation
préliminaire sur un échantillon équilibré de 500 images MNIST. Elle part de
la meilleure de trois solutions non régularisées, choisie par le RV, puis suit
la continuation `eta = 1, 10, 100, 1000`. Les cartes à échelle commune montrent
l'expansion progressive et le compromis avec l'alignement. Elle n'est plus
utilisée dans le manuscrit.

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_regularization_path.py
```

Le script exploratoire volume/UMAP est archivé dans
`archive/exploratory/verify_volume_repulsion.py` ; ses anciens résultats ont été
supprimés et ne font pas partie de ce dossier.

## Balayage hollow et forte régularisation

Le dossier [hollow_regularization_sweep](hollow_regularization_sweep/) contient
un test hors manuscrit de l'objectif hollow régularisé jusqu'à `eta=1000000`.
Toutes les branches partent du même état full-RV. Les cartes à échelle commune
et individuelle, les métriques finales et les trajectoires montrent que l'effet
devient visible vers `eta=3000`, que la cible est presque exactement atteinte à
`eta=100000`, et que `eta=1000000` dégrade davantage l'alignement et la fidélité
locale.

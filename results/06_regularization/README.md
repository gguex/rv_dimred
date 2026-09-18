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

Le script exploratoire volume/UMAP est archivé dans
`archive/exploratory/verify_volume_repulsion.py` ; ses anciens résultats ont été
supprimés et ne font pas partie de ce dossier.

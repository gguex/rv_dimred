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

## Essais à réaliser

L'expérience contrôlée comparant le RV sans pénalité à deux forces de
régularisation sur trois graines reste à réaliser. Ses résultats seront ajoutés
dans un sous-dossier distinct de `math_checks/`, avec les paramètres et le
protocole effectivement utilisés.

Le script exploratoire volume/UMAP est archivé dans
`archive/exploratory/verify_volume_repulsion.py` ; ses anciens résultats ont été
supprimés et ne font pas partie de ce dossier.

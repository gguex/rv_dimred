# Figure conceptuelle de la géométrie des noyaux

Cette figure donne la vue d'ensemble ajoutée au manuscrit révisé. Elle est
entièrement schématique et ne représente pas une expérience numérique.

Les deux panneaux montrent :

1. une coupe du cône de noyaux linéaires réalisables, avec la cible, le plus
   proche noyau de rang contraint et l'angle associé au RV ;
2. une feuille locale de l'image d'un readout non linéaire, avec le gradient
   ambiant et la vitesse tangentielle réellement induite par une mise à jour des
   coordonnées.

Les rayons séparés du premier panneau rappellent que le cône de rang contraint
est généralement non convexe. Le second panneau ne dessine volontairement pas
la mise à jour paramétrique comme une projection orthogonale.

Fichiers produits :

- `kernel_geometry_overview.pdf`, version vectorielle utilisée dans l'article ;
- `kernel_geometry_overview.png`, version de contrôle visuel.

Reproduction depuis la racine du dépôt :

```sh
.venv/bin/python scripts/figures/kernel_geometry_overview.py
```

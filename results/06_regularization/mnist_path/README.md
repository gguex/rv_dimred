# Chemin de régularisation Kernel t-SNE sur MNIST

Cette expérience montre l'effet progressif de la pénalité L² sur la composante
neutre du noyau de sortie. Elle utilise 500 images de MNIST, soit les 50 premières
images disponibles de chaque chiffre dans le fichier de test.

## Protocole

La construction appelée ici « Kernel t-SNE » combine le noyau d'entrée du
protocole t-SNE du manuscrit (affinités gaussiennes adaptatives, perplexité 30,
adoucissement 0,5) avec un noyau de sortie Student-t. L'objectif d'alignement
est le RV complet. Il s'agit donc de la méthode du cadre proposé, pas de la
fonction de perte KL du t-SNE classique.

La pénalité est écrite sous la forme sans dimension

`eta/2 * (rho - rho0)^2`, avec `rho = Tr(K_Y)/(1-sum(f_i^2))`

et une cible intérieure `rho0 = 0.99`. Les cinq niveaux sont
`eta = 0, 1, 10, 100, 1000`.

Trois initialisations PCA légèrement perturbées sont d'abord optimisées sans
régularisation pendant 2 000 pas. La graine 1, qui donne le RV le plus élevé,
est sélectionnée sans consulter les étiquettes, puis raffinée pendant 3 000 pas
supplémentaires. Chaque niveau positif part de la solution du niveau précédent
et utilise 1 000 pas d'Adam. Le pas vaut 0,1. Le calcul est en float32 sur CPU.

## Résultats

| eta | RV | rho | Rayon RMS | ARI | Trustworthiness k=15 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,7008 | 0,9564 | 7,83 | 0,435 | 0,932 |
| 1 | 0,7016 | 0,9573 | 8,07 | 0,447 | 0,932 |
| 10 | 0,7010 | 0,9617 | 9,70 | 0,402 | 0,931 |
| 100 | 0,6924 | 0,9734 | 17,71 | 0,388 | 0,925 |
| 1000 | 0,6611 | 0,9838 | 27,94 | 0,425 | 0,919 |

La figure principale emploie la même échelle pour les cinq cartes. Elle montre
que la hausse de l'inertie du noyau se traduit bien par un étalement progressif
des groupes. Les faibles valeurs changent peu la géométrie. À partir de
`eta=100`, l'expansion est nette et le RV commence à diminuer sensiblement.

Le RV légèrement supérieur pour `eta=1` ne constitue pas une amélioration
générale due à la pénalité. L'objectif est non convexe : la faible perturbation
peut déplacer la continuation vers un bassin dont le RV non pénalisé est un peu
meilleur. Les ARI ne suivent pas une progression monotone et n'ont pas servi au
choix des paramètres. La conclusion défendable reste que la pénalité contrôle
l'inertie et l'étalement, avec un compromis croissant sur l'alignement et la
préservation des voisinages.

## Fichiers

- `mnist_regularization_path.pdf/png` : les cinq cartes à échelle commune et
  le compromis entre inertie et RV ;
- `mnist_regularization_diagnostics.pdf/png` : trajectoires internes de
  l'inertie, du RV et du rayon pour chaque niveau ;
- `summary.csv` : métriques finales ;
- `trajectories.csv` : mesures toutes les dix itérations ;
- `baseline_candidates.csv` : sélection du départ non régularisé ;
- `coordinates.npz` : données, étiquettes, noyau d'entrée et coordonnées ;
- `config.json` : protocole complet, versions et empreintes des sources.

Reproduction depuis la racine du dépôt :

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_regularization_path.py
```

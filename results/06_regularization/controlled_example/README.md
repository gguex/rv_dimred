# Contrôle de la composante neutre : exemple synthétique

Expérience du 18 septembre 2026. Paramètres fixés avant le premier lancement,
sans recherche d'hyperparamètres ni sélection d'initialisation.

## Objectif et protocole

Le solveur maximise le RV complet moins
`lambda * (Tr(K_Y) - t0)^2 / (2*(n-1))`. Cette pénalité est la moitié de la
distance de Frobenius au carré entre la projection neutre de `K_Y` et
`t0*K0/(n-1)`, où `K0 = I - sqrt(f)*sqrt(f)^T`.

Pour rendre les paramètres lisibles, on pose `T = 1 - sum(f_i^2)`,
`rho = Tr(K_Y)/T`, `t0 = rho0*T` et `lambda = eta*(n-1)/T^2`.
L'objectif devient `RV - eta*(rho-rho0)^2/2`.

- 60 objets : trois groupes gaussiens de 20 points dans R³, graine 20260918.
  Centres (-2,0,0), (2,0,1), (0,3,-1), écart-type isotrope 0,6.
  Données centrées, divisées par la distance médiane entre paires distinctes.
- Noyaux d'entrée et de sortie Student-t de paramètre nu=1, avec centrage
  pondéré et poids uniformes. Sortie en dimension 2. RV complet, diagonale incluse.
- Cible illustrative `rho0 = 0.65`, donc `t0 = 0.6391667`.
  Cette cible est un choix de contrôle, pas une valeur optimale apprise.
- Intensités `eta = 0, 1, 10`, soit `lambda = 0, 61.01695, 610.16949`.
- Initialisations `0.1*randn(60,2)` de graines 0, 1 et 2. Pour chaque graine,
  les trois intensités partent exactement des mêmes coordonnées.
- Adam, pas 0,03, 1 500 itérations, sans arrêt anticipé ni sélection d'itéré.
  Calcul CPU en float64, un thread. Mesures toutes les 25 itérations.
- Les étiquettes servent seulement à colorer les figures.

Les paramètres, versions et empreintes du code sont dans [config.json](config.json).
Le score final est recalculé sur les coordonnées retournées par le solveur.
Le rayon RMS est `sqrt(sum_i f_i * ||y_i - mean_f(Y)||²)`.

## Résultats

Moyennes ± écarts-types empiriques sur les trois initialisations :

| eta | RV non pénalisé | Inertie normalisée rho | Erreur absolue à rho0 | Rayon RMS |
|---|---:|---:|---:|---:|
| 0 | 0,99734 ± <0,00001 | 0,42542 ± <0,00001 | 0,22458 ± <0,00001 | 0,72415 ± <0,00001 |
| 1 | 0,99161 ± <0,00001 | 0,57576 ± <0,00001 | 0,07424 ± <0,00001 | 1,11908 ± <0,00001 |
| 10 | 0,96296 ± 0,01435 | 0,64895 ± 0,00640 | 0,00527 ± 0,00093 | 2,04396 ± 0,43318 |

Il faut regarder l'erreur absolue par essai : la moyenne de rho peut masquer
des écarts situés de part et d'autre de la cible.

La pénalité modérée déplace l'inertie vers la cible avec une petite baisse du RV.
Les trois initialisations donnent des métriques presque identiques. La pénalité
forte approche davantage la cible, mais diminue plus le RV et donne des
configurations différentes selon l'initialisation. Les figures montrent des
points éloignés ; avec la graine 1, un groupe se disperse nettement.

Pour eta=0 et eta=1, la norme finale du gradient est au plus 2,5e-8.
Pour eta=10, elle reste entre 1,8e-4 et 4,7e-4 ; l'objectif augmente encore
de 3,6e-5 à 2,4e-4 sur les 100 dernières itérations et les rayons continuent
d'augmenter. Ces essais sont donc des résultats à budget fixé, pas des optima
certifiés. L'expérience ne démontre ni une divergence à l'infini ni une
convergence finale de ces trois trajectoires.

**Conclusion à retenir pour l'article :** la construction contrôle une inertie
dans l'espace des noyaux et permet d'en observer le compromis avec l'alignement.
Une trace proche de la cible ne suffit pas à garantir une bonne configuration
euclidienne. Ne pas présenter cette pénalité comme une amélioration générale
de visualisation ou comme une garantie d'existence d'un optimum fini.

## Fichiers et reproduction

- [summary.csv](summary.csv) : les neuf résultats, traces, RV, rayons et diagnostics.
- [aggregate.csv](aggregate.csv) : moyennes et écarts-types entre graines.
- [trajectories.csv](trajectories.csv) : évolution de chaque essai.
- [coordinates.npz](coordinates.npz) : données, étiquettes, poids, noyau d'entrée,
  trois initialisations et neuf configurations finales.
- [trajectories.pdf](trajectories.pdf) : contrôle d'inertie, RV et étalement.
- [embeddings.pdf](embeddings.pdf) : neuf configurations à la même échelle,
  centrées pour l'affichage, sans alignement des rotations.

```sh
.venv/bin/python scripts/experiments/06_regularization/regularization_run.py
```

Le script régénère les fichiers de résultats de ce dossier avec les paramètres
ci-dessus. Le RV utilise la stabilisation numérique existante du solveur
(`+1e-12` au dénominateur). Le centrage dense existant est conservé.

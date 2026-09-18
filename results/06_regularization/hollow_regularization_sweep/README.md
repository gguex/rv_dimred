# Balayage exploratoire : hollow RV avec régularisation neutre

Cette expérience est un test hors manuscrit. Elle utilise exactement le même
échantillon MNIST (`n=2000`), le même noyau d'entrée et le même état full-RV
initial que la figure de l'article. Chaque essai repart indépendamment de cet
état et maximise le hollow RV avec `eta = 0, 100, 1000, 3000, 10000, 100000,
1000000` pendant 500 pas Adam. La cible est `rho0 = 0.99`.

## Résultats

| eta | rho | erreur à la cible | rayon RMS | hollow RV | trustworthiness |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,9939467 | 0,0039467 | 24,16 | 0,75271 | 0,95260 |
| 100 | 0,9937761 | 0,0037761 | 23,78 | 0,75262 | 0,95214 |
| 1 000 | 0,9922495 | 0,0022495 | 20,96 | 0,74919 | 0,95199 |
| 3 000 | 0,9909756 | 0,0009756 | 19,03 | 0,74541 | 0,95320 |
| 10 000 | 0,9902966 | 0,0002966 | 18,06 | 0,74282 | 0,95293 |
| 100 000 | 0,9900045 | 0,0000045 | 17,13 | 0,73959 | 0,94973 |
| 1 000 000 | 0,9899878 | 0,0000122 | 16,34 | 0,72136 | 0,94003 |

`eta=100` explique peu visuellement : ses valeurs sont presque celles du hollow
non régularisé. L'effet devient clair vers `eta=3000`. Entre `eta=10000` et
`eta=100000`, la pénalité place l'inertie très près de la cible tout en conservant
une structure visuelle proche du hollow. À `eta=1000000`, la pénalité domine,
le hollow RV et la trustworthiness baissent davantage, et l'optimisation devient
plus raide ; cette valeur n'apporte pas de bénéfice pratique ici.

Les cartes à échelle commune montrent surtout la contraction. Les cartes
recadrées séparément montrent que la forme varie progressivement, puis devient
plus compacte à très forte pénalisation.

## Figures

- `embeddings_common_scale.pdf` : sept embeddings sur une échelle commune ;
- `embeddings_autoscaled.pdf` : mêmes embeddings, chacun remplissant son panneau ;
- `final_metrics.pdf` : inertie, erreur à la cible, rayon et hollow RV selon `eta` ;
- `trajectories.pdf` : évolution des sept optimisations.

Les versions PNG, `summary.csv`, `trajectories.csv`, `coordinates.npz` et
`config.json` sont conservées dans le même dossier.

## Reproduction

```sh
.venv/bin/python scripts/experiments/06_regularization/hollow_regularization_sweep.py
```

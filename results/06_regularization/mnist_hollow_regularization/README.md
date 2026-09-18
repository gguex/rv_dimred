# Hollow RV et régularisation sur le même échantillon MNIST

Cette expérience est celle retenue dans le manuscrit révisé. Elle réutilise
exactement les 2 000 observations et les artefacts de l'ancienne figure
`04_tether` : MNIST équilibré, 200 images par chiffre, graine 0, perplexité 30,
adoucissement 0,5 et initialisation PCA.

## Comparaisons

La figure reprend sans modification les coordonnées historiques full-RV et
t-SNE. La solution full-RV après 500 pas constitue l'unique état commun avant
toute modification.

Quatre branches indépendantes partent de ce tableau de coordonnées strictement
identique et utilisent chacune 500 pas Adam : hollow RV, full RV avec `eta=100`,
full RV avec `eta=1000`, et hollow RV avec `eta=100000`. La cible est `rho0=0.99`.
Les deux panneaux d'évolution comparent le hollow, la régularisation forte et
leur combinaison. Ils retiennent les deux quantités directement pertinentes :
l'inertie normalisée dans l'espace des noyaux et le rayon RMS des coordonnées.
Les courbes de RV sont omises, car les branches n'optimisent pas toutes le même
RV et leurs valeurs finales figurent déjà dans les titres.

| Configuration | RV optimisé | rho | rayon RMS | trustworthiness (k=15) |
|---|---:|---:|---:|---:|
| État commun full-RV | 0,67061 | 0,98260 | 12,88 | 0,95505 |
| Hollow RV | 0,75271 | 0,99395 | 24,16 | 0,95260 |
| Full RV, eta=100 | 0,67346 | 0,98631 | 16,80 | 0,95647 |
| Full RV, eta=1000 | 0,67188 | 0,98839 | 20,04 | 0,95502 |
| Hollow RV, eta=100000 | 0,73962 | 0,99000 | 17,13 | 0,94994 |

Le hollow seul pousse l'inertie au-dessus de la cible. La pénalité seule l'en
rapproche par-dessous ; une pénalité plus forte combinée au hollow modère son
expansion et ramène l'inertie vers la cible. Cette expérience illustre les deux
effets à partir du même état, pas une supériorité générale ni une comparaison à
force de pénalisation égale.

## Fichiers

- `mnist_hollow_regularization.pdf` et `.png` : figure complète ;
- `summary.csv` : indices de toutes les configurations ;
- `trajectories.csv` : évolutions hollow et régularisée forte ;
- `coordinates.npz` : données, noyau d'entrée et coordonnées ;
- `config.json` : protocole, versions et empreintes des artefacts historiques.

Reproduction depuis la racine du dépôt :

```sh
.venv/bin/python scripts/experiments/06_regularization/mnist_hollow_regularization.py
```

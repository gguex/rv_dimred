# Log des Expériences (`scripts/test_*.py`)

Résumé de tous les scripts de test exploratoires, **organisés par lien avec les propositions**
de `new_article_plan.md`. Données : MNIST réduit ($n=500$, sauf Test D : $n=50$), perplexité 30
sauf mention, PULL = RV normalisé (cosinus), readout Student-$t$.

## Carte tests → résultats

| Test | Script(s) | Lien plan | Statut |
|------|-----------|-----------|--------|
| A | (recouvrement spectral) | Prop. 3 (plafond) | ✅ coïncidence exacte 0.4294 |
| B | `test_directional_solver.py` | §8 (contre-exemple) | 🔻 démoté (algo non canonique) |
| C | `test_primal_dual.py`, `_grid.py` | Prop. 7 (volume linéaire) | ✅ PUSH réel, dual auto |
| D | `test_manifold_dimension.py` | Prop. 4 (dimension) | ✅ $nd-\binom{d+1}2$, multi-points |
| E | `test_probabilistic_rv.py`, `_grid.py` | Prop. 7 / discussion | ✅ meilleur ARI (0.458) |
| F | `test_log_volume_Kdiag0.py`, `_scan_Kx_diag0.py` | Prop. 6 (hollow) | ✅ hollow-RV pur ARI 0.40 |
| G | `test_diag_energy.py` | Prop. 6 (mécanisme métrique) | ✅ plancher diagonal |

---

## 1. PUSH — la force de répulsion (Prop. 7)

Comment intégrer une répulsion face au PULL canonique (cosinus RV). La répulsion qui marche
agit toujours sur le **Gram brut** $\mathbf G_Y$, jamais sur $\mathbf K_Y$ centré (cf. Prop. 7,
$\mathbf Q\mathbf 1=0$).

### 1.1. Log-volume $\log Z$ — le t-SNE canonique
Le terme $\log Z$ ($Z=$ somme hors-diagonale du Gram Student) engendre la **force répulsive exacte
du t-SNE** (gradient $\propto\sum_j\tilde q_{ij}^2(\mathbf y_i-\mathbf y_j)$).

- **`test_log_volume.py`** : grille (perplexité × $\lambda$). Avec $\lambda\approx0.1$, le cadre RV
  reproduit la géométrie/les clusters de t-SNE, plus robuste topologiquement (meilleur ARI).
- **`test_log_volume_sweep.py`** : balayage rapide de $\lambda$.
- **`test_log_volume_scan.py`** : balayage fin $\lambda\in[0.05,0.3]$, perp 30. Le volume du nuage
  t-SNE est atteint de façon fluide vers $\lambda\approx0.16$. Sert aussi à tester un mauvais noyau
  d'entrée (perplexité non ajustée).
- **Caveat (avec RV plein) :** $\log Z$ non borné ($\approx9$ pour $n=500$) domine le PULL borné
  ($O(1)$) → runaway (spread→50, ARI→0.21). Ne devient *sage* qu'avec le hollow-RV (§2 / Prop. 6).

### 1.2. Primal-dual — volume linéaire (Test C)
Pénalité linéaire sur la masse (au lieu du log non borné).

- **`test_primal_dual.py`** : descente primale-duale, $\lambda$ ajusté par *dual ascent* vers une
  densité cible. PUSH réel (spread $7.6\to19$ monotone), dual auto ($\lambda\to0.18$, ARI 0.42).
- **`test_primal_dual_grid.py`** : grille. Moins tranchant que $\log Z$ (force non logarithmique).

### 1.3. Negative sampling — répulsion ciblée (Test E)
Inspiré d'UMAP : pénalise la proximité des **non-voisins**, $\lambda\langle\mathbf 1-\mathbf G_X,\mathbf G_Y\rangle$.

- **`test_probabilistic_rv.py`** : $\lambda=0$ → ARI 0.374, spread 7.3 ; **$\lambda=1$ → ARI 0.458,
  spread 10.0 (meilleur)** ; $\lambda=5$ → ARI 0.368. Surpasse t-SNE (0.368) sur le **RV plein** —
  la répulsion ciblée n'a pas besoin du hollow. Candidat le plus efficace.
- **`test_probabilistic_rv_grid.py`** : grille (perp × $\lambda$), confirme la compatibilité.

---

## 2. La diagonale du noyau — hollow-RV (Prop. 6)

Mettre à zéro les deux diagonales de $\mathbf K_X,\mathbf K_Y$ avant le cosinus = projection sur
le sous-espace hollow. Plein vs hollow $=$ une seule métrique $\mathbf M=\mathbf I+\mathcal D^*\mathcal D$.

### 2.1. Hollow-RV (Test F)
- **`test_log_volume_Kdiag0.py`** : les deux diagonales à zéro. Hollow-RV pur ($\lambda=0$) :
  **ARI 0.40, spread 12.5** (vs RV plein 0.374 / 7.3) — déjà $>$ t-SNE *sans aucun PUSH*. Puis
  $\log Z$ s'ajoute proprement (spread $12\to38$ monotone, ARI $\sim0.30$–0.40, **pas de runaway**).
  Démontre le rôle de **« global tether »** de la diagonale dans $\|\mathbf K_Y\|$.
- **`test_log_volume_scan_Kx_diag0.py`** : variante asymétrique ($\mathbf K_X$ hollow + stochastique
  par ligne, $\mathbf K_Y$ plein donc tether conservé). Normalisation stochastique → **ARI à froid
  amélioré** (0.41) en raccrochant les points isolés ; peak **ARI 0.44 à $\lambda=0.1$**.

### 2.2. Mécanisme métrique (Test G)
- **`test_diag_energy.py`** : suit `spread`, `frac_diag` $=\|\mathrm{diag}\,\mathbf K_Y\|^2/\|\mathbf K_Y\|^2$,
  et `sum_r2` $=\sum_i r_i^2$ (énergie de degré $=$ diagonale, car $K_{ii}=-r_i$).

  | config | spread | frac_diag | sum_r2 | RV_plein | RV_hollow |
  |---|---|---|---|---|---|
  | full-RV ($\lambda=0$) | 7.3 | 0.14 | 0.0018 | 0.697 | 0.762 |
  | hollow-RV ($\lambda=0$) | 12.5 | 0.36 | 0.0019 | 0.619 | 0.791 |
  | t-SNE sklearn | 19.5 | 0.58 | 0.0020 | 0.478 | 0.764 |

  **Mécanisme (correction d'une hypothèse initiale fausse).** On avait conjecturé que le RV plein
  *gonfle* $\sum_i r_i^2$ → faux : `sum_r2` est un **plancher quasi constant** (0.0018→0.0020). Ce
  qui varie est `frac_diag`, car l'énergie hors-diagonale $\|\mathbf o_Y\|^2$ **s'effondre** quand on
  s'étale. Le RV plein normalisant par ce plancher diagonal, l'étalement est pénalisé → tether
  (spread plafonné 7.3, frac_diag verrouillé 0.14). Le hollow retire le plancher → spread libéré.
  `frac_diag` (0.14 → 0.36 → 0.58) est l'observable du régime (compact/MDS → étalé/SNE).

---

## 3. Géométrie de l'ensemble atteignable

### 3.1. Dimension de la variété (Test D, Prop. 4)
- **`test_manifold_dimension.py`** : rang du jacobien $\mathbf Y\mapsto\mathbf K_Y$ par autograd.
  Confirme $\dim=nd-\binom{d+1}2$ (pour $d=2$ : $2n-3$). Vérifié sur **5 points $\mathbf Y_0$
  aléatoires** par $d$ : rangs $49/97/144$ constants ($d=1,2,3$, $n=50$), falaise spectrale
  $\gtrsim10^{13}$ → dimension **générique** (preuve : Prop. 4, Lemmes A/B).

### 3.2. Solveur directionnel (Test B) — échec instructif
- **`test_directional_solver.py`** : solveur alternatif (sans descente sur $\mathbf Y$) alternant
  (A) inversion du readout (MDS pour retrouver $\mathbf Y$) et (B) déplacement du Gram vers
  $\mathbf K_X$. Effondrement pour Student-$t$ (RV $0.38\to0.057$).
  **Réserve :** l'implémentation testée ne suit pas l'Algorithme 1 canonique → **démoté** en
  remarque conceptuelle (§8 du plan), rendu inutile par Prop. 5 (gradient = riemannien).

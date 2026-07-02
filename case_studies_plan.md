# Case Studies (art. §7) — Plan de rédaction et d'expériences

Compagnon de `new_article_plan.md` (§5–6) pour la section 7 de l'article. Trois principes :

1. **Posture de validation, pas de benchmark.** La section est organisée *par affirmations de la
   théorie* (une expérience = une prédiction vérifiée), dans **l'ordre du récit théorique**
   (cône → variété → gradient → diagonale → dial). Aucune comparaison « fw vs ref » en ARI,
   aucun récit « on bat t-SNE ».
2. **Tout à n = 2000** (MNIST stratifié 200/classe ; single-cell PBMC3k n=2638 tel quel ;
   Swiss roll n=2000). *Exception structurelle* : la dimension de la variété (7.2) se mesure sur
   n ∈ {30, 50, 80(, 120)} — le rang du jacobien est infaisable à n=2000
   (matrice ~2·10⁶ × 4000) et l'argument repose précisément sur la constance du rang *à travers* n.
   Toutes les expériences *données* sont à n=2000.
3. **La vitrine GitHub absorbe les coupes.** Footnote GitHub dès l'intro de la section ; tout
   scatterplot, sweep ou variante coupé du papier va dans la galerie du repo.

Statuts : ✅ résultat existant réutilisable tel quel · 🔁 à refaire (n=2000 / protocole) ·
🔶 nouveau à produire.

---

## 1. Structure de la section (~3–3.5 pages, 2 figures + 3 tables)

### 7.0 Intro + setup (½ page)
« Chaque résultat structurel fait une prédiction mesurable ; nous les vérifions une à une. »
Setup commun compressé : 3 datasets, poids uniformes, q=2, autograd/Adam (500 it., lr 0.1,
init PCA partagée), softening γ=0.5 (§5.3.2 de l'ancien papier, désormais partie de la méthode).
Indices définis en 2–3 lignes chacun, *seulement ceux utilisés* : Procrustes, kNN overlap (k=15),
trustworthiness, spread, fraction diagonale. Footnote GitHub (galerie + reproduction, seeds figés).

### 7.1 Le cône : recouvrement spectral exact et plafond — valide Th. 1–2, Prop. 1, Cor. 1
- **Table 2** (unique artefact) : 6 méthodes (PCA, KPCA-RBF, Isomap, LLE, Diffusion, Laplacian
  Eigenmaps) × 3 datasets, **forme close** (une eigendécomposition) vs librairie de référence.
  Colonnes : Procrustes, kNN, **RV/RV_max = 1.000** (la colonne vedette : rattache la table à la
  Prop. 1). Exception honnête Laplacian Eigenmaps (~10⁻¹, readout orthonormal) expliquée en 2 lignes.
  Remark courte sur le kernel projecteur (LLE/LE), sans la digression de l'ancien papier.
- **Test A en une phrase** : le gradient à sortie linéaire atteint RV_max(2) au chiffre près
  (valeur 🔁 à re-mesurer à n=2000 ; l'ancien 0.4294 était sur MNIST réduit).
- Coupé → vitrine : figure des scatterplots spectraux, colonnes ARI, hyperparamètres détaillés.

### 7.2 La variété : dimension de la nappe — valide Prop. 2 (promesse §4.2)
- **Table 3** (minuscule) : rang(DΦ) = nq − C(q+1,2) pour q ∈ {1,2,3} × n ∈ {30,50,80}
  (✅ existant ; 🔶 option : ajouter n=120 pour la robustesse). Falaise spectrale (~10¹³) en texte.
- Pas de figure (spectre du jacobien → vitrine).

### 7.3 Le gradient : deux forces, une prédiction — valide Cor. 2 + Prop. 6(b) (promesse §4.5) `[PAYOFF]`
- (i) **Une phrase** : les formes analytiques PULL (K̃_X · q̃²) et PUSH ((4τ/Z) · q̃²) coïncident
  avec l'autograd du vrai code à la précision machine (`test_attraction_power.py`, 🔁 relancer à
  n=2000 pour l'uniformité — coût nul).
- (ii) **Table 4** : Procrustes croisés (framework_tSNE, framework_UMAP, reference_tSNE,
  reference_UMAP) × (MNIST, single-cell). Le point : les deux variantes du framework sont plus
  proches *l'une de l'autre* que de leurs références → l'objectif domine le noyau, le cadre
  approche t-SNE et pas UMAP — **exactement la prédiction de la taxonomie §6.2** (PUSH par-paire
  hors d'atteinte). 🔁 refaire à hyperparamètres cohérents avec 7.4 (perplexité/k=30, γ=0.5) —
  les chiffres existants (0.124/0.184 à hp=15) sont à hp différent.
- Coupé → vitrine : side-by-side scatterplots t-SNE/UMAP, sweeps (hp × γ) complets.

### 7.4 La diagonale : tether et libération — valide Prop. 5 + Lemme (promesse §5.3) `[FIGURE SIGNATURE]`
- **Figure 1** (double) : (haut) triptyque full-RV → hollow-RV → t-SNE ; (bas) décomposition
  d'énergie ‖K̊_Y‖² qui s'effondre vs plancher Σr²ᵢ quasi constant, le long de l'optimisation.
- 🔁 **À refaire intégralement à n=2000** (les chiffres actuels — spread 7.3/12.5/19.5,
  frac_diag 0.14/0.36/0.58, plancher 0.0018→0.0020 — sont à n=500). Légende à figer :
  MNIST n=2000, perplexité 30, γ=0.5, seed 0. Le corps de l'article (§5.3) ne cite aucun chiffre
  → seule la légende/texte de §7 porte les valeurs. Ne pas conflater avec le sweep γ (vitrine).

### 7.5 Le dial supervisé : t-SNE ↔ classes — illustre §3.4 (mélange de cibles), « no library counterpart »
- **Figure 2** : courbes train/test (ARI + trustworthiness) vs β sur MNIST et single-cell,
  point d'opération honnête β≈0.5. Le strip de scatterplots d'évolution → vitrine.
- 🔁 **À refaire avec le nouveau protocole** (cf. §5.3.3 réécrit) : dial *unique* sur l'entrée,
  K_β = (1−β)·K_ag + β·K_Z (composantes normées Frobenius avant mélange), sortie Student-t fixe —
  plus la double interpolation entrée+sortie de l'ancien papier. Protocole train/test inchangé
  (70% stratifié, K_Z sur labels train uniquement, placement out-of-sample Gaussien, aucun label
  test utilisé).

**Budget : Figures 1–2 ; Tables 2–4** (Table 1 = taxonomie PUSH, déjà en §6.2). Sous le plafond
« 3 figures max » du plan.

---

## 2. Restructuration du repo

Cible (la refonte se fait *au fil des expériences*, cf. §3 — pas de big-bang) :

```
rv_dimred/
├── src/                          # bibliothèque (inchangée) : rv_kernels, datasets, indices, benchmark_common
├── experiments/                  # ← renomme scripts/ ; UN dossier par case study, numéroté comme §7
│   ├── 01_spectral/              #   spectral_run.py, spectral_indices.py (+ colonne RV/RV_max)
│   ├── 02_manifold_dim/          #   test_manifold_dimension.py (renommé manifold_dim_run.py)
│   ├── 03_forces/                #   test_attraction_power.py (→ forces_check.py) + approximations_finetune_run.py + cross-Procrustes
│   ├── 04_tether/                #   test_diag_energy.py (→ tether_run.py) + figure signature
│   └── 05_supervised_dial/       #   nouveau, en repartant de scripts/old/hybrids_*.py
├── showcase/                     # vitrine : scripts de galerie (scatter grids, sweeps γ/hp, side-by-side)
│   └── gallery/                  #   images finales référencées par le README
├── results/                      # miroir de experiments/ : results/01_spectral/, ..., + results/showcase/
├── archive/                      # tout ce qui ne nourrit ni l'article ni la vitrine (histoire git préservée)
│   ├── old_paper/                #   scripts/old/* (pipelines de l'ancien papier), rv_dimred_old.tex
│   ├── exploratory/              #   test_directional_solver (Test B), test_primal_dual* (Test C),
│   │                             #   test_probabilistic_rv*, test_log_volume* (sauf si repris en vitrine)
│   └── notes/                    #   tests_log.md (+ tests_log2.md), figé ; experiments_notes.md RESTE à la racine (log vivant)
├── data/                         # inchangé
└── README.md                     # devient la vitrine : galerie, une ligne par case study, lien arXiv
```

**Classement article / vitrine / archive des fichiers existants :**

| Existant | Destin | Rôle |
|---|---|---|
| `scripts/spectral_run.py`, `spectral_indices.py` | **article** → `experiments/01_spectral/` | Table 2 |
| `scripts/spectral_figures.py` | **vitrine** → `showcase/` | scatter grids spectraux |
| `scripts/tests/test_manifold_dimension.py` | **article** → `experiments/02_manifold_dim/` | Table 3 |
| `scripts/tests/test_attraction_power.py` | **article** → `experiments/03_forces/forces_check.py` | phrase autograd 7.3(i) |
| *(nouveau)* `experiments/03_forces/cross_procrustes_run.py` | **article** | Table 4 (cross-Procrustes, hp canonique) |
| `scripts/approximations_finetune_run.py` | **vitrine** → `showcase/` (moteur de sweep ; la Table 4 vient du script ciblé) | sweeps γ×hp |
| `scripts/approximations_finetune_figures.py`, `_scatter.py` | **vitrine** → `showcase/` | side-by-side, sweeps |
| `scripts/tests/test_diag_energy.py` | **article** → `experiments/04_tether/` | Figure 1 |
| `scripts/old/hybrids_*.py` | **base de travail** → `experiments/05_supervised_dial/` (adapté), originaux → `archive/old_paper/` | Figure 2 |
| `scripts/tests/test_log_volume*.py` | **archive/exploratory** (démo objectif composite RV − τ log Z reviviscible en vitrine si E6 la retient) | option |
| `scripts/tests/test_primal_dual*.py` | **archive/exploratory** (Test C, corroboration citée en discussion) | — |
| `scripts/tests/test_directional_solver.py` | **archive/exploratory** (Test B, démoli par Prop. 4) | — |
| `scripts/tests/test_probabilistic_rv*.py` | **archive/exploratory** (hors plan) | — |
| `scripts/old/{spectral,approximations,demo,make_readme}*.py` | **archive/old_paper** | — |
| `scripts/tests/tests_log.md` | **archive/notes** (figé ; les faits importants sont dans experiments_notes.md) | — |
| `results/*/old/`, `results/figures/tests/` | **archive** (ou purge, git garde l'historique) | — |

**Hygiène au passage** : ajouter `__pycache__/`, `.ruff_cache/`, `.DS_Store` au `.gitignore` et
les sortir du suivi ; supprimer `scripts/__pycache__/` du disque.

---

## 3. Expériences à faire/refaire — et rangement associé

L'ordre suit §7 ; chaque expérience emporte son étape de rangement (« pendant qu'on y est »),
si bien que la restructuration du §2 est terminée quand la dernière expérience tourne.

**E1 — Spectral + plafond (7.1)** ✅ (fait le 2026-07-02)
- n=2000 confirmé partout (les datasets l'étaient déjà). Colonne RV/RV_max ajoutée
  (`rv_ceiling()` dans src/rv_kernels, Prop. 1) : **ratio = 1.0000 pour les 4 readouts linéaires
  × 3 datasets** (Th. 2 vérifié en forme close) ; readouts orthonormaux sous le plafond comme
  attendu (LLE 0.71–0.95, LE 0.85–0.99 — convention d'axes de la référence, pas l'optimum RV).
- Test A (`ceiling_check.py`, entrée adaptive-Gaussian perp 30, γ=0.5) : **gradient = plafond à
  ~10⁻⁷** — mnist 0.3027, singlecell 0.3770, swissroll 0.2143 (remplacent l'ancien 0.4294).
- Procrustes vs librairies : ≤ 0.007 partout sauf LE/singlecell 0.097 et LE/mnist 0.021
  (l'exception honnête de l'article).
- *Rangement fait* : `experiments/01_spectral/` (run + indices + ceiling_check),
  `showcase/spectral_figures.py`, `archive/old_paper/` (scripts + rv_dimred_old.tex, récupéré
  depuis OneDrive `rv_dimred_3/`), sorties → `results/01_spectral/`, anciens
  `results/{coordinates,indices}/spectral` supprimés (git rm), `results/figures/spectral` gardé
  jusqu'à E6. `.gitignore` était déjà correct. Helpers `exp_*` ajoutés à benchmark_common.

**E2 — Dimension de la variété (7.2)** ✅ (fait le 2026-07-02, avec n=120)
- **rang(DΦ) = nq − C(q+1,2) exactement** pour tout q ∈ {1,2,3} × n ∈ {30,50,80,120}, sur 5 points
  Y0 aléatoires (généricité) — falaise spectrale 10¹³–10¹⁴ (`min_gap`), rang parfaitement défini.
  Headline q=2 : dim = 2n−3 (57/97/157/237). n=120 confirme que la formule n'est pas un accident.
- *Rangement fait* : script → `experiments/02_manifold_dim/manifold_dim_run.py` (n=120 ajouté au
  sweep, sortie CSV `results/02_manifold_dim/indices/manifold_dim.csv`, docstring recalée sur
  art. §7.2 / Prop. 2, lint propre) ; `test_directional_solver.py`, `test_primal_dual*.py`,
  `test_probabilistic_rv*.py` → `archive/exploratory/` ; `tests_log.md` → `archive/notes/`.

**E3 — Forces + cross-Procrustes (7.3)** ✅ (fait le 2026-07-02)
- (i) `forces_check.py` à n=2000 : les trois identités passent à la précision machine —
  **(A) attraction RV = q² à 4·10⁻¹⁹** (et ≠ q¹, écart 1·10⁻⁴ ≫ le match, décisif),
  **(B) gradient t-SNE = (p−q)q¹ à 9·10⁻⁹**, **(C) push −log Z = q² à 4·10⁻¹⁸**. Confirme
  Cor. 2 + Prop. 6(b) : attraction RV et push volumique tous deux en q² (via κ′=−q²), t-SNE
  asymétrique (q¹ / q²).
- (ii) `cross_procrustes_run.py`, matrices 4×4 à perp/k=30, γ=0.5, n=2000. **La prédiction tient
  nettement** : les deux variantes du framework sont plus proches l'une de l'autre que de leurs
  références. MNIST : fw_tsne~fw_umap = **0.122** < fw_tsne~ref_tsne 0.254, fw_umap~ref_umap 0.323 ;
  single-cell : **0.190** < 0.362, 0.470. Robuste vs l'ancien hp=15 (0.124/0.184). Nuance qui
  renforce §6.2 : fw_umap est *plus loin* de ref_umap que fw_tsne de ref_tsne — brancher les noyaux
  UMAP ne rapproche pas d'UMAP, l'objectif domine.
- *Rangement fait* : `test_attraction_power.py` → `experiments/03_forces/forces_check.py` (n=2000,
  CSV) ; nouveau `experiments/03_forces/cross_procrustes_run.py` ; le moteur de sweep
  `approximations_finetune_{run,figures,scatter}.py` → `showcase/` (les sweeps sont vitrine, la
  Table 4 vient du script ciblé) ; `test_log_volume*.py` → `archive/exploratory/` (démo objectif
  composite reviviscible depuis git si E6 la retient). Sorties → `results/03_forces/indices/`.

**E4 — Tether (7.4)** 🔁 **prioritaire** (les chiffres de la légende n'existent pas encore à n=2000)
- Refaire full-RV / hollow-RV / t-SNE à n=2000, perp 30, γ=0.5, seed 0 ; extraire spread,
  frac_diag, et les courbes ‖K̊_Y‖² vs Σr²ᵢ le long de l'optimisation ; composer la Figure 1.
- *Rangement* : `experiments/04_tether/` ; `results/figures/tests/` → archive ; miroir
  `results/04_tether/`.

**E5 — Dial supervisé (7.5)** 🔶 (protocole nouveau)
- Implémenter le dial unique K_β (entrée seulement, sortie Student-t fixe), β ∈ {0,.25,.5,.75,1},
  train/test 70/30, MNIST + single-cell à n=2000 ; produire la Figure 2 (courbes) et le strip
  d'évolution (→ vitrine).
- *Rangement* : `experiments/05_supervised_dial/` en adaptant `hybrids_*.py`, originaux archivés ;
  `results/coordinates/old/` → archive.

**E6 — Vitrine** 🔶 (après E1–E5, tout à n=2000)
- Régénérer la galerie complète : scatter grids spectraux (6×3), side-by-side t-SNE/UMAP,
  strip du dial, sweep γ (mécanique %-entrées-négatives, spread 7→26), spectre du jacobien,
  démo objectif composite (si retenue). Réécrire `README.md` en page vitrine (une image + une
  ligne par case study, lien vers l'article).
- *Rangement final* : supprimer `scripts/` (vide), vérifier que `results/` est un miroir propre,
  tag git `pre-submission`.

---

## 4. Chiffres de l'article à mettre à jour après reruns

- §7.1 : valeur du Test A (ex-0.4294) et ordres de grandeur Procrustes de la Table 2.
- §7.3 : les Procrustes croisés à hp=30 (n=2000) — MNIST fw_tsne~fw_umap 0.122, ~ref_tsne 0.254,
  fw_umap~ref_umap 0.323, ref_tsne~ref_umap 0.152 ; single-cell 0.190 / 0.362 / 0.470 / 0.218
  (remplacent les ex-0.124/0.184/0.294/0.362 à hp=15). Identités des forces : q² à ~10⁻¹⁸.
- §7.4 : spread / frac_diag / plancher (ex-7.3/12.5/19.5, 0.14/0.36/0.58, 0.0018→0.0020 à n=500).
- §7.5 : ARI/trust train-test du nouveau protocole (les valeurs de l'ancien papier — test ARI
  0.36→0.53 MNIST, 0.50→0.89 single-cell — correspondent à l'ancien dial double, ne pas réutiliser).
- Vérifié : les sections théoriques (§4–§6) ne codent aucun chiffre en dur — seuls §7 et les
  légendes portent des valeurs.

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
- **Figure 2** : courbes train/test (ARI + trustworthiness) vs β sur MNIST et single-cell.
  Le strip de scatterplots d'évolution → vitrine.
- ✅ **DÉCISION (2026-07-06) : double dial** — un seul β déplace l'entrée ET la sortie ensemble :
  `K_in(β) = (1−β)·K_ag + β·K_Z` (K_Z sur base **linéaire**) et
  `K_out(β) = (1−β)·StudentT(ν=1) + β·linéaire(Y)` (composantes normées Frobenius avant mélange),
  objectif **RV hollow** (§7.4). C'est le protocole de l'ancien papier, revalidé après un détour
  raté par le « dial unique / sortie Student-t figée » qui cassait la généralisation (test single-cell
  déclinait). Dialer la sortie vers le linéaire est *nécessaire* : seul un noyau de sortie linéaire
  réalise la structure de centroïdes rang-(m−1) de K_Z ; une sortie Student-t figée ne le peut pas.
  β=0 → t-SNE, β=1 → cMDS sur centroïdes (collapse des classes). Protocole train/test inchangé
  (70% stratifié, K_Z sur labels train uniquement, placement out-of-sample Gaussien, aucun label
  test utilisé). **Résultat** : test ARI monte de 0.35→0.54 (MNIST, pic β=0.75) et 0.50→0.92
  (single-cell, β=1) sur points jamais étiquetés ; trustworthiness tenue ≈ t-SNE jusqu'au régime
  intermédiaire ; β=1 sur-apprend (train ARI→1), donc le test récompense un β intermédiaire/quasi-plein.

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

**E4 — Tether (7.4)** ✅ (fait le 2026-07-02)
- full-RV / hollow-RV / t-SNE à n=2000, perp 30, γ=0.5, seed 0. **La prédiction tient** :
  le plancher de degré **Σr²ᵢ est quasi constant** (4.82 / 4.89 / 4.96 ·10⁻⁴ pour full/hollow/t-SNE ;
  3.7·10⁻⁴ → 4.8·10⁻⁴ le long de l'optim), tandis que l'énergie structurelle **‖K̊_Y‖² s'effondre**
  (2.84·10⁻² → 3.84·10⁻³ full ; → 1.96·10⁻³ hollow ; 0.60·10⁻³ t-SNE). Étalement croissant
  12.9 / 16.4 / 35.5 ; frac_diag 0.11 / 0.20 / 0.45 ; ARI 0.42 / 0.40 / 0.53. Le full-RV est
  *tethered* (spread bloqué, ‖K̊_Y‖² plus haut), le hollow libère vers t-SNE.
- **Figure 1** produite (`tether_figure.{png,pdf}`) : triptyque (haut) + panneau d'énergie log (bas,
  ‖K̊_Y‖² qui s'effondre vs Σr²ᵢ épinglé). Légende figée : MNIST n=2000, perp 30, γ=0.5, seed 0.
- *Rangement fait* : `test_diag_energy.py` → `experiments/04_tether/tether_run.py` (n=2000, pipeline
  canonique gaussian_affinity_base+soften_and_center, trajectoires + CSV) + nouveau `tether_figure.py` ;
  sorties → `results/04_tether/{coordinates,indices}/` + figure à la racine de `results/04_tether/` ;
  `results/figures/tests/` → `archive/exploratory/figures/`, `results/figures/old/` →
  `archive/old_paper/figures/`. `scripts/tests/` est désormais vide (reste `scripts/` vide → purge à E6).

**E5 — Dial supervisé (7.5)** ✅ (double dial, décidé 2026-07-06)
- Double dial (entrée K_in(β)=(1−β)K_ag+βK_Z, K_Z base linéaire ; sortie K_out(β)=(1−β)StudentT+β·linéaire),
  objectif RV hollow, β ∈ {0,.25,.5,.75,1}, train/test 70/30, MNIST + single-cell à n=2000.
  Scripts : `supervised_dial_run.py`, `supervised_dial_indices.py`, `supervised_dial_figure.py` (Figure 2),
  `dial_scatter_figure.py` (grille d'évolution, `results/05_supervised_dial/dial_scatter.png`).
- Résultat : test ARI 0.35→0.54 (MNIST) / 0.50→0.92 (single-cell) sur points jamais étiquetés ;
  β=1 sur-apprend (train ARI→1). Le détour « dial unique / Student-t figé » a été testé et rejeté
  (single-cell déclinait) — cf. §7.5.
- *Rangement* : `experiments/05_supervised_dial/` (fait), originaux `hybrids_*.py` dans
  `archive/old_paper/scripts/`, `results/coordinates/old/` → archive (fait).

**E6 — Vitrine** 🔶 en cours (après E1–E5, tout à n=2000)
- ✅ `README.md` réécrit en page vitrine : résumé de l'article, galerie (spectral_gallery + tether
  + dial_scatter + courbes dial), quickstart vérifié (snippet exécuté : linear rv = ceiling = 0.6792,
  student-t hollow OK), table de repro par case study, layout, citation. Nouvelle figure
  `showcase/readme_gallery.py` → `results/figures/spectral_gallery.png` (3 datasets × 6 méthodes).
- 🔶 Reste : galerie complète optionnelle (side-by-side t-SNE/UMAP, sweep γ, spectre jacobien),
  purge de `scripts/` (vide), vérifier que `results/` est un miroir propre, tag git `pre-submission`.

---

## 4. Chiffres de l'article à mettre à jour après reruns

- §7.1 : valeur du Test A (ex-0.4294) et ordres de grandeur Procrustes de la Table 2.
- §7.3 : les Procrustes croisés à hp=30 (n=2000) — MNIST fw_tsne~fw_umap 0.122, ~ref_tsne 0.254,
  fw_umap~ref_umap 0.323, ref_tsne~ref_umap 0.152 ; single-cell 0.190 / 0.362 / 0.470 / 0.218
  (remplacent les ex-0.124/0.184/0.294/0.362 à hp=15). Identités des forces : q² à ~10⁻¹⁸.
- §7.4 : spread 12.9/16.4/35.5, frac_diag 0.11/0.20/0.45, plancher Σr²ᵢ 3.7·10⁻⁴→4.8·10⁻⁴ le long
  de l'optim (≈constant à 4.9·10⁻⁴ entre configs), ‖K̊_Y‖² 2.84·10⁻²→3.84·10⁻³ (full) à n=2000
  (remplacent ex-7.3/12.5/19.5, 0.14/0.36/0.58, 0.0018→0.0020 à n=500).
- §7.5 : ARI/trust train-test du double dial (n=2000, 70/30) — test ARI 0.353→0.536 MNIST (pic β=0.75),
  0.501→0.917 single-cell (β=1) ; train ARI→1.000 à β=1 (sur-apprentissage) ; trust test tenue ~0.82–0.85.
  (Proches de l'ancien papier 0.36→0.53 / 0.50→0.89 : c'est le même double dial, revalidé.)
- Vérifié : les sections théoriques (§4–§6) ne codent aucun chiffre en dur — seuls §7 et les
  légendes portent des valeurs.

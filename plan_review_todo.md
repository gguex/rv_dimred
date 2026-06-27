# Plan review — open follow-ups

Reliquat de la relecture détaillée de `new_article_plan.md` (2026-06-27). Les 3 findings de
fond + 5 triviales sont **déjà appliquées** (voir « Résolu » ci-dessous) ; ce qui suit est le
backlog restant, non bloquant. *Les numéros de ligne dérivent à chaque édition — se repérer par
section / nom de Proposition.*

## Résolu (2026-06-27, pour mémoire)
- **#1** attraction `q̃²` (cadre) vs `q̃¹` (t-SNE) — corollaire Prop. 5 réécrit + asymétrie PULL/PUSH ;
  test `scripts/tests/test_attraction_power.py` ; entrée `experiments_notes.md` (2026-06-27).
- **#2** taxonomie PUSH / UMAP — réécrite (global-`Z` vs par-paire, place du `log`) ; refs McInnes (UMAP),
  Damrich & Hamprecht.
- **#8** nouveauté Prop. 6 — recentrée sur la métrique `M=I+D*D` (pas « diagonale = degré ») ; attribution
  Laplacien Chung / von Luxburg.
- **#3** collision `λ` (push) vs `γ` (softening) ; **#4** « quatre contributions » ; **#5** double `[SIGNATURE]` ;
  **#9** `=2n−3` exact ; **#10** convention de numérotation §N (plan) vs `art. §N` (article).

---

## A. Cohérences à fermer

### #6 — Test C : « démoté » mais cité comme preuve de deux Props signature
- **Où** : §6 table des tests le renvoie en `🔻 DISCUSSION → §8` ; or Prop. 6 (paragraphe « tether »,
  *« RV plein ⇒ … le PUSH volume effondre (Test C) »*) et Prop. 7 (réserves, *« Appui empirique : Test C
  (volume réel) »*) s'appuient dessus.
- **Problème** : si Test C étaie les deux pièces signature, ce n'est pas qu'un appendice — tension de statut.
- **Fix** : préciser *quelle* mesure de Test C est citée, et soit la promouvoir en évidence (petite table/figure),
  soit reformuler l'appui de Prop. 6/7 pour ne pas s'appuyer sur un test « démoté ». ~30 min rédaction.

### #11 — Prop. 5 : « mêmes points critiques » seulement sur 𝒰
- **Où** : Prop. 5 (iii) + paragraphe « Honnêteté (à écrire tel quel) ».
- **Problème** : l'équivalence `∇_Y f = 0 ⟺ grad F = 0` ne vaut que pour `Y ∈ 𝒰` (points engendrant
  affinement ℝ^d) ; les configurations effondrées (points coïncidents) sont des points critiques de bord
  exclus — à dire pour rester honnête.
- **Fix** : une phrase dans le paragraphe « Honnêteté ». ~10 min.

## B. Reproductibilité

### #7 — Chiffres de la figure (b) à épingler
- **Où** : Prop. 6 (mécanisme tether) `spread 7.3 → 12.5`, `frac_diag 0.14 → 0.36 → 0.58`, **MNIST n=500** ;
  repris en §5(b) et §6 (Test G).
- **Problème** : (i) le `spread 12.5` (hollow) dépend fortement de `γ` — `experiments_notes.md` montre
  `7.4 → 26.3` sur mnist quand `γ : 0.25 → 1.0` — donc le « 12.5 » n'est reproductible qu'à `γ` fixé ;
  (ii) Test G est à **n=500**, le sweep des notes à **n=2000** → ne pas conflater dans le papier.
- **Fix** : figer `dataset + n + γ` pour le run de la figure (b) et l'écrire dans la légende ; vérifier que les
  chiffres tiennent à ce `γ`. ~1 h (re-run + relevé).

## C. Renforcements bon marché

### #12 — Rang de Prop. 4 à plusieurs `n`
- **Où** : `scripts/tests/test_manifold_dimension.py` balaie `d=1,2,3` mais seulement **n=50**.
- **Problème** : un relecteur peut soupçonner une coïncidence à n=50.
- **Fix** : ajouter une boucle `n ∈ {30, 50, 80}` et montrer `rang = nd − C(d+1,2)` à chaque fois.
  ~20 min (rerun, le test existe déjà).

### #13 — Figure (a) plus riche (table spectrale déjà calculée)
- **Où** : §5(a) « Recouvrement spectral exact » ne montre que « le gradient atteint le plafond ».
- **Opportunité** : la vitrine spectrale (`scripts/spectral_run.py` → `results/indices/spectral/`) a déjà
  produit **toute une table** de méthodes (PCA / Isomap / KPCA / LE / LLE) atteignant *chacune* son
  `RV_max(d)` — figure bien plus riche pour le corollaire d'unification (Th. 2). **Données déjà là**, coût
  quasi nul (script de figure à écrire). ~1 h.

### #14 — Énoncer « RV = CKA centré » tôt
- **Où** : art. §2 (ou §1). Cortes (2012, CKA) déjà cité (réf. §10).
- **But** : préempter le réflexe « ce n'est pas juste CKA ? ». Une phrase : RV = cosinus de Frobenius de
  noyaux centrés = CKA ; la nouveauté n'est pas le cosinus mais la *géométrie de l'ensemble atteignable*.
  ~10 min.

### #15 — Vérifier l'énoncé d'équivalence Moran avant de l'affirmer
- **Où** : art. §2, positionnement *« le RV linéaire coïncide avec la maximisation d'autocorrélation de
  Moran / MEM / MULTISPATI »* (cité comme antérieur : Bavaud, Dray et al.).
- **Problème** : si l'énoncé exact n'est pas dans ces refs tel quel, c'est une responsabilité factuelle.
- **Fix** : relecture ciblée de Bavaud / Dray ; caler la formulation précise, ou rétrograder « coïncide avec »
  → « apparenté à ». Coût : une lecture (~1-2 h selon accès aux papiers).

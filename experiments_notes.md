# Experiment notes

Running log of empirical findings worth remembering (not a formal write-up).

---

## 2026-06-26 — Why the framework approximates t-SNE but not UMAP

**Setup.** `approximations_finetune` sweep: hollow-RV objective, framework =
adaptive-Gaussian / fuzzy input kernel + Student-t / UMAP output kernel, swept over
(perplexity|n_neighbors) × lambda (input-affinity softening exponent γ). Reference =
sklearn t-SNE / umap-learn with the same neighbour hyperparameter.

**Observation.** Across all datasets the framework reproduces t-SNE noticeably better
than UMAP (lower Procrustes, higher kNN overlap to the reference).

**Diagnosis — the objective dominates, not the kernels.** We only swap the input/output
*kernels*; the *objective* (RV-cosine, hollow) is the same for both. Procrustes at
(hp=15, λ=0.5):

| | mnist | singlecell |
|---|---|---|
| framework_UMAP vs reference_UMAP | 0.326 | 0.422 |
| framework_UMAP vs **framework_tSNE** | **0.124** | **0.184** |
| framework_tSNE vs reference_tSNE | 0.294 | 0.362 |
| reference_UMAP vs reference_tSNE (the two libraries) | 0.055 | 0.219 |

The two framework variants are far closer to *each other* than to their respective
references: plugging in UMAP's kernels does **not** move us toward UMAP — it keeps us at
"the RV-cosine embedding". The objective shapes the result; the kernel swap is secondary.

**The missing dynamic = UMAP's loss.** UMAP minimises an *unnormalised* fuzzy-set binary
cross-entropy with a *per-pair* repulsive term `(1-p_ij) log(1-q_ij)` (negative sampling),
with no global partition function Z. The RV-cosine has no analog of either: its global
`‖K_Y‖` normalisation and zero-sum centering repulsion make it a structural cousin of
**t-SNE** (global-Z normalisation + KL attraction), so the t-SNE family is "in range" of
the cosine while UMAP is not.

**Consistent with our own theory (Prop. 7 / PUSH taxonomy).** UMAP's PUSH lives on the
*raw* Gram G_Y with a log-rational g; the RV-cosine is volume-blind (no raw-Gram push). So
the framework itself predicts UMAP is unreachable without leaving the pure cosine and
adding a repulsion term on G_Y (negative sampling / −λ log Z / BCE) — the "Plan 2 / solver"
extension, not the RV cadre. Reproducing UMAP is a missing *mode*, not a missing knob.

**Secondary factors (minor):** the output-kernel a,b parameters; and forcing a PCA init on
umap-learn (overriding its default spectral init), which already makes the reference
somewhat atypical.

### Related: lambda (input softening γ) and the "push" without a volume term

The spread that appears as λ→1 is **not** a t-SNE volume repulsion. Every centered kernel
obeys K√f = 0 (zero weighted row-sums), so attraction on neighbour pairs is exactly
balanced by **negative (repulsive) off-diagonal entries** for non-neighbours — a zero-sum
push-pull baked in by the double-centering. λ=γ controls the *contrast* of the target:
γ=1 (no softening) gives a peaked affinity → after centering ~94% of off-diagonal entries
are negative → strong spread; γ=0.25 fills in the affinity → ~68% negative, low contrast →
compact. The hollow-RV removes the diagonal "tether" that otherwise suppresses this spread.
Measured on mnist (perplexity=30): spread 7.4 → 26.3 and %negative off-diagonal 67.6 →
94.2 as λ goes 0.25 → 1.0. The push is genuine but **bounded** (matches a fixed target),
not a runaway volume term.

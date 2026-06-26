# RV Dimensionality Reduction — Plan resserré (cible JMVA)

Plan de **convergence vers soumission** de l'article
*« The Kernel Inner Product Space: A Unified Framework for Dimensionality Reduction »*.
On resserre : moins de prospection, un récit théorique net, une partie expérimentale minimale.

Statut : ✅ dérivé + vérifié · 🟡 à formaliser (matériel prêt) · 🔶 à dériver (faisable) ·
🔴 ouvert / hors-scope.

> **Décision de cadrage (ce plan).** Pièce signature = **Prop. 6** (la diagonale = métrique de
> degré). On *ne* met *pas* au premier plan « on reproduit / on bat t-SNE » : c'est contre-productif
> pour JMVA et ça invite une revue de type benchmark. Ce qui sort du corps de l'article est listé en
> §8 (pour mémoire, avec raisons).

---

## 0. La narration (le fil)

> **La réduction de dimension est un alignement de cosinus (RV) dans un espace de noyaux pondérés
> $\mathcal K_n$ ; la *géométrie de l'ensemble atteignable* — cône PSD pour une lecture linéaire,
> variété mince et courbe pour une lecture à queue lourde — caractérise pourquoi certaines méthodes
> sont spectrales (forme close) et d'autres itératives, et une *unique décomposition algébrique* (la
> diagonale du noyau = métrique de degré) explique la frontière MDS ↔ neighbor-embedding.**

Arc en trois temps :
1. **Le cadre** — un objet ($\mathcal K_n$, cosinus RV), une question : *quels noyaux $\mathbf K_Y$
   une sortie peut produire, et lequel maximise l'alignement avec $\mathbf K_X$ ?*
2. **La dichotomie géométrique** — la nature de l'ensemble atteignable (**cône** vs **variété**)
   *est* l'explication du zoo des méthodes. Le cœur.
3. **La clé algébrique** — la diagonale du noyau est le curseur entre les deux régimes (Prop. 6).
   Le **hook neuf** qui distingue le papier d'un énième « kernel view of DR ».

---

## 1. Les trois contributions à vendre (et l'argument de nouveauté)

**Risque dominant pour JMVA : la nouveauté.** La « vue noyau » de la réduction de dimension existe
déjà (Ham 2004 ; CKA/Cortes 2012 ; Bengio). L'intro doit répondre frontalement à *« qu'y a-t-il de
neuf au-delà du kernel view ? »* par les trois contributions, *toutes* absentes de la littérature
kernel-view :

1. **Caractérisation exacte de l'ensemble atteignable** : cône PSD de rang $\le d$ (linéaire,
   Th. 1) vs **variété de dimension $2n-3$** (non linéaire, Prop. 4). Pas « c'est un noyau » : *quels*
   noyaux, et de quelle forme géométrique.
2. **Plafond d'alignement en forme close** $\mathrm{RV}_{\max}(d)$ (Prop. 3) — analogue Frobenius de
   la variance expliquée, quantité neuve et interprétable.
3. **La diagonale comme métrique de degré** (Prop. 6) — full-RV vs hollow-RV $=$ une seule métrique
   $\mathbf M=\mathbf I+\mathcal D^*\mathcal D$ ; curseur MDS ↔ SNE. *Personne ne l'a écrit.*
4. **Aveuglement au volume et origine de la répulsion** (Prop. 7) — $\mathbf Q\mathbf 1=0\Rightarrow$
   la RV ignore le volume $Z$ ; la répulsion t-SNE *est* le gradient de ce mode jeté. Lecture
   **forme/taille** (statistique de forme) — transforme le caveat d'échelle en théorème, très JMVA.

---

## 2. Cadre vs solveur : deux plans (à garder bref mais explicite)

- **Plan 1 — le cadre (la contribution) :** $\mathcal K_n$, RV comme objectif, factorisation
  entrée/sortie. Une *formulation*, indépendante du solveur.
- **Plan 2 — le solveur :** gradient / optimisation riemannienne. Au Plan 2, l'unification MM de
  t-SNE/UMAP existe déjà (Yang et al.) → on n'y va pas en force (cf. §8).

Conséquence : notre unification est au **Plan 1** (même problème, même $\mathcal K_n$), donc plus
fondamentale que celle des solveurs. Le solveur paramétrique standard est *re-caractérisé*
(gradient = gradient riemannien, Prop. 5), pas remplacé.

---

## 3. Structure de l'article (~25–30 p.)

1. **Introduction** `[réorienter]` — public stat : analyse multivariée (Escoufier, diagramme de
   dualité) ∪ manifold learning. Les 3 contributions de §1, formulées comme théorèmes.
2. **L'espace $\mathcal K_n$** `[polir]` — $\mathbf Q=\bm\Pi^{1/2}\mathbf H$, double-centrage,
   dualité MDS ; RV = cosinus ; cône PSD $\mathcal K_n^+$, projection de Higham. **Positionnement
   (attribué, ~3 phrases, pas un résultat) :** lu avec $\mathbf K_X$ comme opérateur de lag, l'objectif
   RV *linéaire* coïncide avec la maximisation d'autocorrélation de Moran / Moran Eigenvector Maps /
   MULTISPATI — connu (Bavaud ; Dray et al.), cité comme **antérieur**. Sert seulement à parler la
   langue JMVA et à interpréter $\mathbf K_X$ ; aucune revendication sur le linéaire.
3. **Réduction linéaire = projection sur un cône** `[étendre]` — Th. 1, Th. 2, Prop. 3 ; corollaire
   d'unification spectrale (PCA/MDS/Isomap/KPCA/LLE/LE).
4. **Au-delà du cône : la variété** `[nouveau]` — Prop. 4 (dim $2n-3$) ; I.2 comme **proposition**
   (gradient = projection tangente), *la* justification du solveur paramétrique.
5. **La diagonale comme métrique de degré** `[nouveau — section signature]` — Prop. 6 ; referme la
   dichotomie §3↔§4 (diagonale = signal sur le cône, artefact sur la variété).
6. **Attraction–répulsion : forme et taille** `[nouveau — résultat, pas simple discussion]` —
   Prop. 7 : $\mathcal K_n$ aveugle au volume, la répulsion comme gradient du mode jeté, lecture
   forme/taille. Relié à Prop. 6 (le hollow lève le tether, le volume sépare) et à la dichotomie
   cône/variété. Lien MM / Yang et al. mentionné en *programme*.
7. **Illustrations** `[~2–3 pages]` — voir §5.
8. **Conclusion** `[bref]` — objectifs composites ; **perspective ouverte** (seul morceau Moran neuf) :
   les neighbor embeddings comme *autocorrélation spatiale non-linéaire à taille contrôlée* (extension
   du cadre Moran/MEM, linéaire, à la variété via Prop. 4 + Prop. 7) — posé comme question, pas résultat.

---

## 4. Résultats théoriques à rédiger (le contenu mathématique)

Notations : $\mathbf{K}=\mathbf{Q}\mathbf{G}\mathbf{Q}^\top$, $\mathbf{Q}=\bm\Pi^{1/2}\mathbf{H}$,
$\mathbf{H}=\mathbf{I}-\mathbf{1}\mathbf{f}^\top$, $\mathcal{K}_n=\{\mathbf{K}\ \text{sym.}:\mathbf{K}\sqrt{\mathbf f}=0\}$,
$\dim\mathcal{K}_n=\binom{n}{2}$. RV = cosinus de Frobenius. (Remarque utile : $K_{ij}=\sqrt{f_if_j}\,B_{ij}$
avec $\mathbf B=\mathbf H\mathbf G\mathbf H^\top$ — le Frobenius plein sur $\mathbf K$ *est déjà* le
produit pondéré par les masses ; pas de double choix de métrique à faire.)

### ✅ Théorème 1 — Ensemble atteignable (sortie linéaire)
Pour $\mathbf{G}_Y=\mathbf{Y}\mathbf{Y}^\top$, $\mathbf{Y}\in\mathbb{R}^{n\times d}$,
l'ensemble des noyaux atteignables est **exactement**
$$\mathcal{S}_d=\{\mathbf{K}\in\mathcal{K}_n:\mathbf{K}\succeq 0,\ \mathrm{rang}(\mathbf{K})\le d\}.$$
Bijection : pour toute cible $\mathbf{K}=\mathbf{U}\bm\Lambda\mathbf{U}^\top$ de rang $\le d$,
$\mathbf{Y}=\bm\Pi^{-1/2}\mathbf{U}\bm\Lambda^{1/2}$ la réalise et est automatiquement
centrée ($\mathbf{f}^\top\mathbf{Y}=\sqrt{\mathbf f}^\top\mathbf{U}\bm\Lambda^{1/2}=0$ car
$\mathbf{U}\perp\sqrt{\mathbf f}$). Géométrie : cône algébrique non convexe (intersection
cône PSD ∩ variété déterminantielle rang $\le d$).

### ✅ Théorème 2 — Projection en forme close
Sur un **cône**, maximiser le cosinus = projection de Frobenius (optimiser l'échelle :
pour $\hat{\mathbf{K}}$ unitaire, $\min\|\mathbf{K}_X-t\hat{\mathbf{K}}\|^2=\|\mathbf{K}_X\|^2-\langle\mathbf{K}_X,\hat{\mathbf{K}}\rangle^2$).
La projection de $\mathbf{K}_X=\sum_i\lambda_i\mathbf{u}_i\mathbf{u}_i^\top$ (décroissant) sur
$\mathcal{S}_d$ est la **troncature aux $d$ premiers vecteurs propres positifs** :
$$\mathbf{K}_Y^\star=\sum_{j=1}^{d}\max(\lambda_j,0)\,\mathbf{u}_j\mathbf{u}_j^\top,\qquad
\mathbf{Y}^\star=\bm\Pi^{-1/2}\mathbf{U}_d\bm\Lambda_d^{1/2}.$$
= **MDS classique** (Eckart–Young–Mirsky, version PSD). Optimum **global**, sans optima
locaux. *Corollaire* : PCA, MDS, Isomap, Kernel PCA, LLE, Laplacian Eigenmaps sont des
instances (seul le noyau d'entrée change). C'est *pourquoi* ces méthodes sont spectrales
et en un coup.

### ✅ Proposition 3 — Plafond d'alignement en dimension $d$
$$\mathrm{RV}_{\max}(d)=\sqrt{\frac{\sum_{j=1}^{d}(\lambda_j^+)^2}{\sum_i\lambda_i^2}},
\qquad \lambda_j^+=\max(\lambda_j,0).$$
Analogue de la « variance expliquée » de la PCA, mais en métrique de Frobenius (carrés
de valeurs propres, car RV = cosinus de Hilbert–Schmidt). Quantité interprétable, neuve.
**Validé numériquement (Test A)** : le gradient à sortie linéaire atteint exactement ce
plafond (sur MNIST réduit : $\mathrm{RV}_{\max}(2)=0.4294$, atteint au chiffre près).

### ✅ Proposition 4 — Variété (sortie non linéaire) `[dérivée + vérifiée]`
Avec un readout non linéaire $\kappa_y$, l'ensemble atteignable
$\mathcal{S}_d^\kappa=\{\mathbf{Q}\,\kappa(\mathbf{D}^2(\mathbf{Y}))\,\mathbf{Q}^\top:\mathbf{Y}\in\mathbb{R}^{n\times d}\}$
est l'image d'une application non linéaire : variété courbe, **non convexe, plus un cône**
(scaler $\mathbf{Y}$ ne scale plus $\mathbf{K}_Y$). Dimension (modulo invariances de centrage
+ groupe euclidien) $\approx nd-\binom{d+1}{2}$, soit **$2n-3$ pour $d=2$** — nappe mince
dans $\mathcal{K}_n$ ($\sim n^2/2$).
**Dimension vérifiée numériquement (Test D)** : le rang du jacobien $\mathbf{Y}\mapsto\mathbf{K}_Y$
vaut exactement $nd-\binom{d+1}{2}$ pour $d=1,2,3$ ($49,97,144$ à $n=50$), avec une falaise
spectrale nette ($\sim10^{14}$) — confirmé sur plusieurs points $\mathbf Y_0$ aléatoires (rang
constant ⇒ générique).
**Énoncé formel.** Soit $\kappa\in C^1([0,\infty))$ avec $\kappa'(t)\neq0$ pour tout $t$ (Student-$t$ :
$\kappa(t)=(1+t)^{-1}$, $\kappa'=-(1+t)^{-2}$). Soit $\Phi:\mathbb R^{n\times d}\to\mathcal K_n$,
$\Phi(\mathbf Y)=\mathbf Q\,\kappa(\mathbf D^2(\mathbf Y))\,\mathbf Q^\top$ avec
$\mathbf D^2(\mathbf Y)_{ij}=\|\mathbf y_i-\mathbf y_j\|^2$. Soit
$\mathcal U=\{\mathbf Y:\text{les }n\text{ points engendrent affinement }\mathbb R^d\}$ (ouvert dense,
$n\ge d+1$). Alors pour tout $\mathbf Y\in\mathcal U$ :
- (i) $\ker D\Phi(\mathbf Y)=\mathcal I_{\mathbf Y}:=\{\mathbf 1\mathbf v^\top+\mathbf Y\mathbf A^\top:\mathbf v\in\mathbb R^d,\ \mathbf A\in\mathfrak{so}(d)\}$
  (mouvements euclidiens infinitésimaux), $\dim=\binom{d+1}2$ ;
- (ii) $\operatorname{rang}D\Phi(\mathbf Y)=nd-\binom{d+1}2$, **constant** sur $\mathcal U$ ;
- (iii) par le **théorème du rang constant**, $\mathcal S_d^\kappa$ est, au voisinage de chaque
  $\Phi(\mathbf Y)$, une **sous-variété lisse plongée** de $\mathcal K_n$ de dimension
  $nd-\binom{d+1}2$ ($=2n-3$ si $d=2$) ;
- (iv) pour $\kappa$ **bornée** (Student-$t$), $\mathcal S_d^\kappa$ est bornée donc **n'est pas un
  cône** (contraste : readout linéaire $\Rightarrow$ cône non borné).

**Preuve.** *Factorisation* $\Phi=C\circ R\circ S$ : $S(\mathbf Y)=\mathbf D^2(\mathbf Y)$,
$R(\mathbf E)=\kappa(\mathbf E)$ terme-à-terme, $C(\mathbf G)=\mathbf Q\mathbf G\mathbf Q^\top$. On a
$DS(\mathbf Y)[\dot{\mathbf Y}]_{ij}=2(\mathbf y_i-\mathbf y_j)^\top(\dot{\mathbf y}_i-\dot{\mathbf y}_j)$
(symétrique, **creuse** : diagonale nulle) ; $DR=\kappa'(\mathbf D^2)\odot(\cdot)$ est un isomorphisme
(multiplication terme-à-terme par des facteurs $\neq0$) ; $C$ est linéaire.

*Lemme A — $\ker D\Phi=\ker DS$.* $\dot{\mathbf Y}\in\ker D\Phi\iff\kappa'(\mathbf D^2)\odot\mathbf W\in\ker C$
où $\mathbf W=DS[\dot{\mathbf Y}]$. Or $C(\mathbf G)=0\iff\mathbf H\mathbf G\mathbf H^\top=0$, et
$\ker C\cap\mathrm{Sym}=\{\mathbf 1\mathbf a^\top+\mathbf a\mathbf 1^\top:\mathbf a\in\mathbb R^n\}$,
de diagonale $2a_i$. Mais $\mathbf W$ est creuse et $\kappa'(0)\neq0$, donc
$\kappa'(\mathbf D^2)\odot\mathbf W$ est creuse ; l'égaler à $\mathbf 1\mathbf a^\top+\mathbf a\mathbf 1^\top$
force $\mathbf a=0$, puis $\kappa'\odot\mathbf W=0$, puis $\mathbf W=0$. (Le centrage ne perd donc
*aucune* dimension ici : les différentielles EDM sont creuses et ne rencontrent $\ker C$ qu'en $0$.)

*Lemme B — $\ker DS=\mathcal I_{\mathbf Y}$ (rigidité infinitésimale du graphe complet).*
$\supseteq$ : $\mathbf D^2$ est invariant sous $\mathbf Y\mapsto\mathbf Y\mathbf R^\top+\mathbf 1\mathbf t^\top$
($\mathbf R\in O(d)$), donc en dérivant $DS[\mathbf Y\mathbf A^\top]=DS[\mathbf 1\mathbf v^\top]=0$ ;
de plus $\mathbf 1\mathbf v^\top+\mathbf Y\mathbf A^\top=0\Rightarrow\mathbf A(\mathbf y_i-\mathbf y_j)=0$,
et les différences engendrant $\mathbb R^d$ on a $\mathbf A=0,\mathbf v=0$, d'où $\dim\mathcal I_{\mathbf Y}=\binom{d+1}2$.
$\subseteq$ : soit $(\mathbf y_i-\mathbf y_j)^\top(\mathbf z_i-\mathbf z_j)=0\ \forall i,j$ ($\mathbf z=\dot{\mathbf Y}$).
On centre $\mathbf u_i=\mathbf y_i-\bar{\mathbf y}$, $\mathbf w_i=\mathbf z_i-\bar{\mathbf z}$ (la
condition ne dépend que des différences). En sommant sur $j$ à $i$ fixé, avec $\sum\mathbf u=\sum\mathbf w=0$ :
$n\,\mathbf u_i^\top\mathbf w_i+\sum_j\mathbf u_j^\top\mathbf w_j=0$, donc $\mathbf u_i^\top\mathbf w_i$ est
constant ; en resommant sur $i$, cette constante est nulle : $\mathbf u_i^\top\mathbf w_i=0\ \forall i$.
La condition devient $\mathbf u_i^\top\mathbf w_j+\mathbf u_j^\top\mathbf w_i=0$, i.e. $\Sigma:=\mathbf U^\top\mathbf W$
**antisymétrique** ($\mathbf U=[\mathbf u_i],\mathbf W=[\mathbf w_i]\in\mathbb R^{d\times n}$). Comme les
$\mathbf u_i$ engendrent $\mathbb R^d$, $\mathbf U$ est de rang $d$, donc $\mathbf W\mapsto\mathbf U^\top\mathbf W$
est **injectif** ; l'espace des $\Sigma$ antisymétriques à colonnes dans $\mathrm{lig}(\mathbf U)\cong\mathbb R^d$
est de dimension $\binom d2$, atteint par $\mathbf W=\mathbf A\mathbf U$, $\mathbf A\in\mathfrak{so}(d)$.
Donc $\mathbf w_i=\mathbf A\mathbf u_i$, soit $\mathbf z_i=\mathbf A\mathbf y_i+\mathbf v$
($\mathbf v=\bar{\mathbf z}-\mathbf A\bar{\mathbf y}$) : $\dot{\mathbf Y}\in\mathcal I_{\mathbf Y}$.

*Conclusion.* Lemmes A+B ⇒ $\ker D\Phi(\mathbf Y)=\mathcal I_{\mathbf Y}$, $\dim=\binom{d+1}2$, d'où
(i)–(ii) ; le rang étant constant sur l'ouvert $\mathcal U$, le théorème du rang constant donne (iii).
(iv) : $\kappa$ bornée ⇒ $\|\kappa(\mathbf D^2)\|_F\le n$ ⇒ $\|\Phi(\mathbf Y)\|\le\|\mathbf Q\|_{\mathrm{op}}^2\,n$
borné ; un cône non trivial est non borné. $\qquad\square$

*Remarque (lien dichotomie).* Le Lemme A est l'exact pendant « volume » de Prop. 6/7 : le centrage est
inoffensif sur les différentielles **creuses** (EDM) car son noyau exige une diagonale non nulle. La
**bornitude** de $\mathcal S_d^\kappa$ (iv) est la cause profonde du besoin de répulsion (Prop. 7) :
contrairement au cône linéaire (non borné, échelle libre), la variété Student-$t$ est confinée, et
séparer les amas exige de réinjecter le mode volume hors-$\mathcal K_n$.

### ✅ Proposition 5 — Le gradient paramétrique comme gradient riemannien (ex-I.2)
Justifie le solveur paramétrique standard : ce que PyTorch calcule sur $\mathbf Y$ *est* l'optimisation
riemannienne sur $\mathcal S_d^\kappa$, à un préconditionnement près.

**Énoncé.** Soit $F:\mathcal K_n\to\mathbb R$ lisse, $\mathbf G=\nabla F(\mathbf K_Y)\in\mathcal K_n$ son
gradient ambiant (Frobenius), et $f=F\circ\Phi$ l'objectif vu dans la carte $\mathbf Y$. Pour
$\mathbf Y\in\mathcal U$ (Prop. 4), posons $T=T_{\mathbf K_Y}\mathcal S_d^\kappa=\mathrm{Im}\,D\Phi(\mathbf Y)$
et $P_T$ la projection orthogonale sur $T$ dans $\mathcal K_n$. Alors :
- (i) **gradient paramétrique** : $\nabla_{\mathbf Y}f=(D\Phi(\mathbf Y))^{*}\,\mathbf G$ ;
- (ii) **gradient riemannien** de $F|_{\mathcal S_d^\kappa}$ (métrique induite) : $\operatorname{grad}F=P_T\,\mathbf G$ ;
- (iii) **mêmes points critiques** : $\nabla_{\mathbf Y}f=0\iff P_T\mathbf G=0\iff\mathbf G\perp T$ ; et le
  pas paramétrique est une remontée de $F$ le long de la variété :
  $\langle D\Phi[\nabla_{\mathbf Y}f],\mathbf G\rangle=\|\nabla_{\mathbf Y}f\|^2\ge0$, avec
  $D\Phi[\nabla_{\mathbf Y}f]=D\Phi\,D\Phi^{*}\mathbf G\in T$. La descente sur $\mathbf Y$ réalise donc une
  remontée riemannienne **préconditionnée** par la métrique tirée-en-arrière
  $\mathbf M_{\mathbf Y}=D\Phi^{*}D\Phi$ ($P_T$ vs $D\Phi D\Phi^{*}$ : même image $T$, mêmes zéros).

**Pour la RV.** $F=\langle\hat{\mathbf K}_X,\cdot\rangle/\|\cdot\|$ ($\hat{\mathbf K}_X=\mathbf K_X/\|\mathbf K_X\|$)
donne, à un facteur $>0$ près, $\mathbf G\propto\mathbf R:=\mathbf K_X-\beta\mathbf K_Y$ avec
$\beta=\langle\mathbf K_X,\mathbf K_Y\rangle/\|\mathbf K_Y\|^2$ (le **résidu structurel**, échelle moindres
carrés). Ainsi : *« gradient structurel » (résidu $\mathbf R$ dans $\mathcal K_n$) et « gradient
paramétrique » (chain rule $\mathbf Y\mapsto\mathbf K_Y$) sont la même flèche*, l'un projeté sur $T$,
l'autre tiré dans la carte — exactement la projection tangente du résidu.

**Preuve.** (i) Chain rule : $Df[\dot{\mathbf Y}]=\langle\mathbf G,D\Phi[\dot{\mathbf Y}]\rangle=\langle D\Phi^{*}\mathbf G,\dot{\mathbf Y}\rangle$,
donc $\nabla_{\mathbf Y}f=D\Phi^{*}\mathbf G$. (ii) Définition du gradient riemannien d'une sous-variété
plongée (Prop. 4 donne que $T$ en est l'espace tangent) : $\operatorname{grad}F=P_T\nabla F=P_T\mathbf G$.
(iii) $D\Phi^{*}\mathbf G=0\iff\mathbf G\perp\mathrm{Im}\,D\Phi=T\iff P_T\mathbf G=0$ ; et
$\langle\mathbf G,D\Phi D\Phi^{*}\mathbf G\rangle=\|D\Phi^{*}\mathbf G\|^2$. $\square$

**Corollaire (force PULL explicite, lien Prop. 7).** Avec $\tilde{\mathbf K}_X=\mathbf Q^\top\mathbf K_X\mathbf Q$
et le readout Student-$t$, le terme d'attraction du gradient paramétrique s'écrit
$$\nabla_{\mathbf y_k}\langle\mathbf K_X,\mathbf K_Y\rangle=4\sum_j(\tilde K_X)_{kj}\,(1+d_{kj}^2)^{-2}\,(\mathbf y_j-\mathbf y_k),$$
soit **exactement la force attractive de t-SNE** (poids affinité-d'entrée $\times\,q_{kj}^2$). Couplé à
Prop. 7 (le PUSH $=-\lambda\nabla\log Z=$ répulsion t-SNE), on récupère la structure attraction–répulsion
complète, *avec l'attraction propre au cadre* (alignement RV) au lieu de l'entropie croisée.

**Honnêteté (à écrire tel quel).** Le gradient paramétrique n'est **pas** identique au gradient
riemannien : la carte $\mathbf Y$ n'est pas une isométrie, d'où le préconditionnement $\mathbf M_{\mathbf Y}$.
Ce n'est pas un défaut — c'est la distinction standard « gradient riemannien dans une carte » vs « dans la
variété plongée ». La revendication défendable est : *même variété, même espace tangent, mêmes points
critiques, même caractère de remontée* ; le solveur autograd est donc l'implémentation canonique (et
préconditionnée) de l'optimisation riemannienne, pas un hack.

### ✅ Proposition 6 — La diagonale comme métrique de degré ; full-RV vs hollow-RV `[SIGNATURE]`
**Coordonnées libres de $\mathcal{K}_n$.** Pour $\mathbf{K}\in\mathcal{K}_n$, écrire
$\mathbf{K}=\mathrm{diag}(\mathbf d)+\mathbf o$ ($\mathbf d\in\mathbb R^n$ diagonale,
$\mathbf o\in\mathrm{Hollow}(n)$). La contrainte de centrage $\mathbf{K}\sqrt{\mathbf f}=0$ est
équivalente à $\mathbf d=\mathcal D\mathbf o$ où
$(\mathcal D\mathbf o)_i=-\tfrac1{\sqrt{f_i}}\sum_{j\neq i}o_{ij}\sqrt{f_j}$.
*Donc la diagonale d'un noyau centré est une fonction linéaire déterministe du hors-diagonal* :
$\mathbf o$ est un système de coordonnées **libre et complet** de $\mathcal{K}_n$
($\dim=\binom n2$), et $\mathrm{diag}$ n'apporte aucune information indépendante.

**Identité exacte (le cœur de la Proposition).** Comme $\mathrm{Diag}\perp\mathrm{Hollow}$ sous
Frobenius, pour $\mathbf{K}_X,\mathbf{K}_Y\in\mathcal{K}_n$ :
$$\langle\mathbf{K}_X,\mathbf{K}_Y\rangle=\langle\mathbf o_X,\mathbf o_Y\rangle+\langle\mathcal D\mathbf o_X,\mathcal D\mathbf o_Y\rangle=\langle\mathbf o_X,\mathbf o_Y\rangle_{\mathbf M},\qquad \mathbf M=\mathbf I+\mathcal D^{*}\mathcal D,$$
d'où
$$\boxed{\ \mathrm{RV}_{\text{plein}}=\cos\nolimits_{\mathbf M}(\mathbf o_X,\mathbf o_Y),\qquad
\mathrm{RV}_{\text{hollow}}=\cos\nolimits_{\mathbf I}(\mathbf o_X,\mathbf o_Y).\ }$$
**La seule différence entre RV plein et RV hollow est la métrique** $\mathbf M$ vs $\mathbf I$ sur
l'espace partagé des affinités de paires. $\mathbf M$ ne dépend que du centrage, pas du readout.

**Forme explicite de $\mathbf M$ (poids uniformes).** $(\mathcal D\mathbf o)_i=-r_i$ avec
$r_i=\sum_{j\neq i}o_{ij}$ (degré / centralité), $(\mathcal D^{*}\mathbf v)_{ij}=-\tfrac12(v_i+v_j)$,
$(\mathcal D^{*}\mathcal D\,\mathbf o)_{ij}=\tfrac12(r_i+r_j)$, et
$$\|\mathbf o\|_{\mathbf M}^2=\|\mathbf o\|^2+\sum_i r_i^2,\qquad
\langle\mathbf o_X,\mathbf o_Y\rangle_{\mathbf M}=\langle\mathbf o_X,\mathbf o_Y\rangle+\sum_i r_i(X)\,r_i(Y).$$
Le RV plein ajoute donc le terme $\sum_i r_i^2$ : l'**énergie de degré** (= énergie diagonale,
car $K_{ii}=-r_i$).

**Cas général (poids quelconques $\mathbf f$, $\mathbf F=\mathrm{diag}(\mathbf f)$).** Définir le
**degré pondéré** $\mathbf r=\mathbf o\sqrt{\mathbf f}$, i.e. $r_i=\sum_{k\neq i}o_{ik}\sqrt{f_k}$.
La contrainte $\mathbf K\sqrt{\mathbf f}=0$ donne $\mathcal D\mathbf o=-\mathbf F^{-1/2}\mathbf r$
($d_i=K_{ii}=-r_i/\sqrt{f_i}$). En symétrisant le coefficient de $o_{ij}$ (paire ordonnée) dans
$\langle\mathcal D\mathbf o,\mathbf v\rangle=-\sum_{i\neq j}\tfrac{\sqrt{f_j}}{\sqrt{f_i}}v_i o_{ij}$,
l'adjoint (Frobenius sur Hollow, euclidien sur $\mathbb R^n$) est
$$(\mathcal D^{*}\mathbf v)_{ij}=-\tfrac12\Big(\tfrac{\sqrt{f_j}}{\sqrt{f_i}}v_i+\tfrac{\sqrt{f_i}}{\sqrt{f_j}}v_j\Big),\qquad
(\mathcal D^{*}\mathcal D\,\mathbf o)_{ij}=\tfrac12\Big(\tfrac{\sqrt{f_j}}{f_i}r_i+\tfrac{\sqrt{f_i}}{f_j}r_j\Big)\quad(i\neq j).$$
La métrique est alors
$$\|\mathbf o\|_{\mathbf M}^2=\|\mathbf o\|^2+\mathbf r^\top\mathbf F^{-1}\mathbf r=\|\mathbf o\|^2+\sum_i\frac{r_i^2}{f_i},\qquad
\langle\mathbf o_X,\mathbf o_Y\rangle_{\mathbf M}=\langle\mathbf o_X,\mathbf o_Y\rangle+\sum_i\frac{r_i(X)r_i(Y)}{f_i},$$
et l'énergie diagonale vaut $\sum_i K_{ii}^2=\mathbf r^\top\mathbf F^{-1}\mathbf r$ (PSD, rang $\le n$ ;
$\mathbf M\succeq\mathbf I\succ0$, donc $\cos_{\mathbf M}$ bien défini). Le facteur $1/f_i$ **amplifie
la cohérence de degré des points de faible masse**. *Limite uniforme* $f_i\equiv1/n$ :
$r_i=\rho_i/\sqrt n$ ($\rho_i=\sum_k o_{ik}$), $(\mathcal D^{*}\mathcal D\mathbf o)_{ij}=\tfrac12(\rho_i+\rho_j)$,
$\sum_i r_i^2/f_i=\sum_i\rho_i^2$ — on retrouve la forme ci-dessus.

**Dichotomie cône / variété — quand $\mathbf M$ est justifiée.**
- *Sortie linéaire (cône).* $\mathrm{diag}(\mathbf G_Y)_i=\|y_i\|^2$ : la diagonale du Gram **brut**
  est un vrai signal radial, et $\mathbf M$ (= Frobenius plein) est la métrique dans laquelle
  maximiser le cosinus sur le cône PSD $=$ Eckart–Young (Th. 1–2, optimum global fermé). Garder
  la diagonale est **correct**.
- *Sortie distance-based (variété).* $\mathrm{diag}(\mathbf G_Y)=\kappa(0)\mathbf 1$ est
  **constante** : le readout n'encode l'information que dans le hors-diagonal. La diagonale de
  $\mathbf{K}_Y$ est alors un *artefact de centrage* ; aucun théorème de projection ne rend
  $\mathbf M$ bénigne sur la variété courbe.

**Le mécanisme du tether (vérifié, Test G).** Décomposer le dénominateur :
$\|\mathbf{K}_Y\|^2=\|\mathbf o_Y\|^2+\|\mathbf d_Y\|^2$ avec $\|\mathbf d_Y\|^2=\sum_i r_i^2$.
Numériquement (MNIST n=500), $\sum_i r_i^2$ est un **plancher quasi constant** ($\approx0.0018$
de full-RV à t-SNE) tandis que $\|\mathbf o_Y\|^2$ **s'effondre** quand l'embedding s'étale
(affinités Student-$t\to0$). Le RV plein normalisant par ce plancher, l'étalement est pénalisé
dès qu'il fait chuter $\|\mathbf o_Y\|$ : d'où un spread plafonné (7.3) et une fraction d'énergie
diagonale $\|\mathbf d_Y\|^2/\|\mathbf{K}_Y\|^2$ verrouillée (0.14). Le hollow-RV retire le plancher
$\Rightarrow$ spread libéré (12.5) et fraction diagonale qui rejoint le régime t-SNE (0.36 vs 0.58).
*La correction $\sum_i r_i^2$ de $\mathbf M$ agit donc comme une normalisation-plancher
spread-insensible (un « global tether »), pas comme une cible que l'optimiseur gonflerait.*

**Énoncé à publier.** Full-RV et hollow-RV coïncident à la métrique près
($\mathbf M=\mathbf I+\mathcal D^{*}\mathcal D$ vs $\mathbf I$). $\mathbf M$ est data-justifiée et
exactement résolue (Eckart–Young) sur le **cône linéaire**, où la diagonale porte les rayons
$\|y_i\|^2$ ; elle est non justifiée et frustrante sur la **variété non linéaire**, où la diagonale
du readout est constante ($\kappa(0)$) et n'agit qu'en plancher de normalisation qui interdit
l'étalement. Le hollow-RV est le cosinus fidèle à l'information — purement hors-diagonale — que le
readout non linéaire encode réellement. **Vérifié : Test F (esthétique) + Test G (mécanisme).**

### ✅ Proposition 7 — Aveuglement au volume et origine de la répulsion (ex-I.3) `[SIGNATURE]`
Transforme le caveat d'invariance d'échelle en *résultat* : **pourquoi** la RV ne sépare pas autant que
t-SNE, et **d'où** vient la répulsion. Posons le volume $Z(\mathbf Y)=\mathbf 1^\top\mathbf G_Y\mathbf 1=\sum_{ij}(1+d_{ij}^2)^{-1}$.

**Énoncé.**
- (a) **$\mathcal K_n$ est aveugle au volume.** $\mathbf K_Y=\mathbf Q\mathbf G_Y\mathbf Q^\top$ est invariant
  sous $\mathbf G_Y\mapsto\mathbf G_Y+\alpha\mathbf 1\mathbf 1^\top$ ($\forall\alpha$), tandis que
  $Z\mapsto Z+\alpha n^2$. Donc $\mathbf K_Y$ — et toute fonction de $\mathbf K_Y$ seul, dont la RV —
  ne porte **aucune** information sur $Z$.
- (b) **La répulsion est le gradient du mode jeté = celle de t-SNE.** Le pas d'ascension du terme
  $-\lambda\log Z$ sur $\mathbf y_i$ vaut $\dfrac{4\lambda}{Z}\sum_j\tilde q_{ij}^2(\mathbf y_i-\mathbf y_j)$
  ($\tilde q_{ij}=(1+d_{ij}^2)^{-1}$) — **exactement** la force répulsive de t-SNE.
- (c) **Aucune répulsion possible depuis $\mathcal K_n$.** Tout objectif $F(\mathbf K_Y)$ a une dérivée
  nulle le long de la famille volumique $\mathbf G_Y+\alpha\mathbf 1\mathbf 1^\top$ ; séparer les amas
  exige un terme fonction du **Gram brut** $\mathbf G_Y$ non invariant sous ce mode (p.ex. $Z$).

**Preuve.** (a) $\mathbf Q\mathbf 1=\mathbf F^{1/2}\mathbf H\mathbf 1=\mathbf F^{1/2}(\mathbf 1-\mathbf 1\,\mathbf f^\top\mathbf 1)=\mathbf 0$
(car $\mathbf f^\top\mathbf 1=1$), donc $\mathbf Q(\alpha\mathbf 1\mathbf 1^\top)\mathbf Q^\top=\alpha(\mathbf Q\mathbf 1)(\mathbf Q\mathbf 1)^\top=\mathbf 0$ ;
et $\mathbf 1^\top(\mathbf G_Y+\alpha\mathbf 1\mathbf 1^\top)\mathbf 1=Z+\alpha n^2$. La fibre du centrage
au-dessus de $\mathbf K_Y$ contient donc une droite entière de volumes distincts. *(Plus finement :
$\ker$ du centrage $\cap\,\mathrm{Sym}=\{\mathbf 1\mathbf a^\top+\mathbf a\mathbf 1^\top\}$, sur lequel
$Z=2n\,\mathbf 1^\top\mathbf a$ est non trivial — le volume vit exactement dans ce que le centrage jette.)*
(b) $Z=\sum_{kl}\tilde q_{kl}$, $\partial Z/\partial\mathbf y_i=-4\sum_j\tilde q_{ij}^2(\mathbf y_i-\mathbf y_j)$ ;
d'où $-\lambda\,\partial(\log Z)/\partial\mathbf y_i=(4\lambda/Z)\sum_j\tilde q_{ij}^2(\mathbf y_i-\mathbf y_j)$,
qui pousse $\mathbf y_i$ **loin** de $\mathbf y_j$. Identique à t-SNE : $\mathrm{KL}=\text{const}-\sum_{ij}p_{ij}\log\tilde q_{ij}+\log Z$,
donc t-SNE $=$ attraction entropie-croisée $+\log Z$ ; **notre cadre $=$ attraction RV $+\log Z$**
(même répulsion, attraction différente). (c) Conséquence directe de (a) : $\frac{d}{d\alpha}F(\mathbf Q(\mathbf G_Y+\alpha\mathbf 1\mathbf 1^\top)\mathbf Q^\top)=0$. $\square$

**(c′) Lecture forme / taille (cadre JMVA).** La RV capture la **forme** (configuration invariante
d'échelle, $\mathbf K_Y$ modulo échelle) ; $Z$ est une mesure (monotone) de **taille/compacité** ; les
neighbor embeddings $=$ alignement de forme $+$ contrôle de taille. C'est la décomposition *shape vs
size* (Kendall ; Dryden & Mardia), réinterprétée : la répulsion réinjecte la dimension de taille que
le cosinus de forme évacue par construction.

**Emboîtement (le diptyque centrage).** Prop. 7 complète l'anatomie du centrage : Lemme A de Prop. 4
(inoffensif sur le creux), Prop. 6 (diagonale redondante *dans* $\mathcal K_n$), Prop. 7 (mode volume
jeté *hors* $\mathcal K_n$). Et il s'articule avec Prop. 6 : RV plein $\Rightarrow$ l'attraction sature
et le PUSH volume effondre (Test C) ; hollow-RV $\Rightarrow$ tether levé, le PUSH volume sépare
proprement. Enfin il referme la dichotomie : la **bornitude** de $\mathcal S_d^\kappa$ (Prop. 4(iv)) rend
la contre-force *nécessaire* (queue lourde, affinités saturantes), là où le cône linéaire (non borné)
n'en a pas besoin.

**Corollaire (taxonomie de la famille PUSH).** Le choix du PUSH se paramètre par **deux axes
orthogonaux** : le *domaine* de normalisation ($\mathbf K_Y$ centré vs Gram brut $\mathbf G_Y$) et la
*forme* $g$ appliquée à la masse $Z$. (Pour la discussion §6 — situe le cadre vs les méthodes.)

| méthode | domaine | $g$ | $\lambda=g'(Z)$ | PUSH |
|---|---|---|---|---|
| **RV (ce cadre)** | $\mathbf K_Y$ **centré** | quadratique | — (mode volume nul) | **non** (redistribution à somme nulle) |
| Elastic Embedding | $\mathbf G_Y$ **brut** | linéaire | const. | oui, constant |
| t-SNE / SNE | $\mathbf G_Y$ **brut** | $\log Z$ | $1/Z$ | oui, auto-ajustant |
| UMAP | $\mathbf G_Y$ brut | log-rationnel | $\partial_Z g$ | oui |

Le RV ne répulse pas pour *deux* raisons cumulées : domaine (normalisation sur $\mathbf K_Y$ centré, de
composante $\mathbf 1\mathbf 1^\top$ nulle) **et** forme (auto-pénalité quadratique à somme nulle). Le
choix de $g$ sur $\mathbf G_Y$ sélectionne le membre (linéaire → elastic embedding ; $\log$ → t-SNE).

**Réserves (à écrire).** C'est le **PUSH**, pas le PULL ; succès pratique conditionné à Prop. 6 ;
$\log Z$ non borné (réglage de $\lambda$) ; optimisation locale. Appui empirique : Test C (volume réel),
Test F/G (comportement avec hollow-RV).

---

## 5. Illustrations (3 figures max — validations de théorèmes)

- (a) **Recouvrement spectral exact** : le RV gradient-linéaire atteint le plafond Prop. 3
  (Test A). 1 table (Procrustes, RV) + courbe/plafond.
- (b) **Géométrie diagonale / hollow** : full-RV (compact, `frac_diag` 0.14) → hollow-RV (étalé,
  0.36) → t-SNE (0.58), avec spread et `frac_diag` qui co-varient (Tests F+G). *Meilleure figure :
  elle illustre Prop. 6, la pièce signature.*
- (c) *optionnel* : **dimension de la variété** $\approx 2n-3$ (Test D, rang du jacobien).

Datasets : Swiss-roll (variété connue) + MNIST réduit ou single-cell. Tout le reste (grilles de
scatterplots, multi-datasets, comparaisons librairies, dial supervisé) → annexe ou coupé (§8).

---

## 6. Tests / preuves de concept

| Test | Objectif | Statut / rôle |
|------|----------|---------------|
| **A** | Plafond linéaire $\to\mathrm{RV}_{\max}(d)$ (Prop. 3) | ✅ → **figure (a)** — coïncidence exacte (0.4294) |
| **D** | Dimension intrinsèque, rang jacobien $\approx2n-3$ (Prop. 4) | ✅ → **figure (c) optionnelle** — $49,97,144$ ($n=50$), falaise $\sim10^{14}$ |
| **F** | Hollow-RV vs RV plein (Prop. 6) | ✅ → **figure (b)** — hollow-RV pur ARI 0.40, étalement libéré |
| **G** | Mécanisme métrique (Prop. 6) : $\sum_i r_i^2$ = plancher | ✅ → **figure (b)** (`test_diag_energy.py`, `tests_log2.md`) — `frac_diag` 0.14→0.36→0.58 suit le spread |
| **B** | Solveur directionnel | 🔻 **DEMOTE** → §8 (remarque conceptuelle, pas figure) |
| **C, E** | Primal-dual / negative sampling | 🔻 **DISCUSSION** seulement → §8 |

**Test D — détail.** Sur $N=50$, jacobien de $\mathbf{Y}\mapsto\mathbf{K}_Y$ via autograd, aplati en
$\binom{N}{2}\times(Nd)$, rang numérique (seuil sur valeurs singulières). Attendu
$Nd-\binom{d+1}{2}=2N-3$ pour $d=2$ ($97$ à $N=50$), confirmant la minceur de la variété
($\sim N^2/2=1225$). Détails empiriques complets : `tests_log.md`, `tests_log2.md`.

---

## 7. Chemin critique vers soumission (ordre conseillé)

1. **Rédiger Th. 1–2 + Prop. 3** — matériel prêt (Test A), pur travail d'écriture.
2. ✅→📝 **Prop. 5 (gradient = gradient riemannien)** — énoncé + preuve faits (adjoint $D\Phi^*$,
   mêmes points critiques, préconditionnement $\mathbf M_{\mathbf Y}$) ; corollaire force PULL = attraction
   t-SNE. Reste : mise au propre + une phrase sur le préconditionnement. Rend le solveur directionnel inutile.
3. ✅→📝 **Prop. 4** — énoncé + preuve faits (rang constant via Lemmes A/B, non-cône par bornitude) ;
   vérifié (Test D, multi-points). Reste : mise au propre rédactionnelle.
4. **Finir Prop. 6** — algèbre faite (adjoint inclus) ; reste : lemme « $\mathbf M$ data-justifiée
   ⟺ readout linéaire » + régularité $\mathbf M\succ0$. Rédiger la section signature §5(article).
5. ✅→📝 **Prop. 7 (volume → répulsion)** — énoncé + preuve faits ($\mathbf Q\mathbf 1=0$, gradient
   $-\log Z=$ répulsion t-SNE, forme/taille). Reste : mise au propre + intégration au §6.
6. **Positionnement Moran** — ~3 phrases en §2 (attribué : Bavaud, MEM, MULTISPATI) + 1 perspective
   non-linéaire en conclusion. *Pas* un résultat ; aucune dérivation à faire.
7. **Réduire les figures à 3** (a/b/c ci-dessus) ; tout le reste en annexe ou coupé.
8. **Intro + positionnement nouveauté** (§1) — le levier d'acceptation n°1.
9. **Discussion** — Prop. 7 (§6) ; MM (Yang et al.) en programme, honnête et bref.

---

## 8. Ce qu'on a OMIS du corps de l'article (pour mémoire)

Décisions de cadrage assumées. Le matériel reste dans le dépôt (`tests_log.md`, `tests_log2.md`,
scripts) ; il n'entre simplement pas comme résultat principal.

- **Théorème 5 — méta-algorithme MM unifiant.** *Description :* présenter spectral, t-SNE/UMAP et
  l'alignement RV comme instances d'un schéma de majoration-minimisation (différant par la raideur
  $k_{ij}$). *Raison de l'omission :* niveau recherche, non dérivé ; **chevauche Yang et al.
  (2014–15)** → nouveauté contestable ; rabbit-hole. → une phrase en discussion (« programme »).
  **Pépite à garder pour cette phrase :** pour le PUSH volumique $g(Z)$, le multiplicateur
  $\lambda=g'(Z)$ est *à la fois* la variable duale (lecture primal-dual de Prop. 7) et la pente
  tangente du pas de majoration MM (si $g$ concave). Ce « pas dual = pente MM » relie Prop. 7 à la
  famille de Yang et al. et la *différencie* (discrépance RV = nouveau membre, dégénéré côté PUSH).

- **Solveur directionnel + Test B.** *Description :* résoudre l'incorporation par projection alternée
  en espace de noyaux (inversion readout ↔ MDS rang-$d$) ; effondrement observé pour Student-$t$
  (RV $0.38\to0.057$). *Raison :* (i) l'implémentation testée **ne suit pas l'Algorithme 1 exact**
  (réserve d'honnêteté) → dangereux à publier sans re-run ; (ii) **devenu inutile** une fois Prop. 5
  (gradient = riemannien) en place. → remarque conceptuelle d'un paragraphe : « hors du cône, la
  projection de Frobenius cesse d'être valide ; la projection alternée tombe hors variété », sans en
  faire une figure.

- **Negative sampling + benchmarks « on bat t-SNE » (Test E, et variantes de PUSH).** *Description :*
  répulsion par échantillonnage négatif $\langle\mathbf 1-\mathbf G_X,\mathbf G_Y\rangle$ (ARI 0.458 >
  t-SNE), pénalité de volume linéaire, sweeps de $\lambda$. *Raison :* « battre t-SNE en ARI » est une
  posture ML que JMVA ne valorise pas et qui invite une revue benchmark ; variantes redondantes une
  fois Prop. 7 en place. → mention d'une phrase / annexe. **NB :** la *dérivation* volume → répulsion
  n'est **plus** omise — elle est promue en **Prop. 7** (le mode volume $\log Z$ = répulsion t-SNE) ;
  seul l'habillage benchmark sort.

- **Le récit « on reproduit / on bat t-SNE ».** *Raison :* non rigoureux, affaiblit un papier de
  théorie et attire les mauvais relecteurs. Garder uniquement comme *illustration* de Prop. 6.

- **Hollow-RV comme objectif de remplacement.** *Raison :* il casse la théorie spectrale
  (Eckart–Young, Th. 1–2). On garde le **RV plein canonique** ; le hollow est présenté comme
  *spécialisation* neighbor-embedding (Prop. 6), pas comme nouvel objectif. *(Question laissée
  ouverte : faut-il même centrer dans ce régime, ou un cosinus d'affinités hollow brutes ≈ CKA non
  centré ? — hors-scope du papier actuel.)*

- **Dial supervisé $\beta$, benchmarks multi-datasets, grilles de scatterplots, comparaisons
  exhaustives de librairies.** *Raison :* tangentiels au récit géométrique ; JMVA veut des théorèmes,
  pas un battle de benchmarks. → annexe minimale ou coupés.

---

## 9. Positionnement / avertissements

- L'unification MM de t-SNE/UMAP **existe déjà** (Yang et al.) — ne jamais présenter comme « nous
  découvrons que t-SNE est un MM ».
- **Nouveauté à défendre frontalement vs Ham (2004) / CKA (Cortes 2012)** : ce n'est pas « la vue
  noyau », c'est la *caractérisation géométrique de l'ensemble atteignable* + le *plafond* + la
  *métrique de degré* (§1).
- Honnêteté assumée sur (i) la non-convexité intrinsèque de la variété (localité), (ii) la RV cosinus
  invariante d'échelle ⇒ pas de fonction de partition ⇒ séparation inter-clusters un peu moindre que
  t-SNE (caveat I.3). *Les relecteurs stat récompensent l'honnêteté* — contrairement aux venues ML.

---

## 10. Références clés

Déjà dans l'article : Robert & Escoufier (1976, RV/duality), Schölkopf (1998, KPCA),
Ham (2004, kernel view), Higham (1988, PSD projection), Barshan (2011, supervised PCA),
Sugiyama (2007, LFDA), Cortes (2012, CKA), Fouss (2005, commute-time), Bavaud (2024).

À ajouter :
- **Optimisation riemannienne (Prop. 5)** : Absil, Mahony & Sepulchre — *Optimization Algorithms
  on Matrix Manifolds* (2008) ; Boumal — *An Introduction to Optimization on Smooth Manifolds* (2023).
- **Solveur / MM (Plan 2, discussion)** : Yang, Peltonen, Kaski — ICML 2014 ; AISTATS 2015 ;
  Carreira-Perpiñán — *Elastic Embedding*, ICML 2010 ; Vladymyrov & Carreira-Perpiñán —
  *Partial-Hessian*, ICML 2012 ; Böhm, Berens, Kobak — *Attraction-Repulsion Spectrum*, JMLR 2022 ;
  de Leeuw — SMACOF / Guttman.
- **EDM (remarque solveur directionnel)** : Dattorro ; Krislock & Wolkowicz — *Euclidean Distance Matrices*.
- **Autocorrélation spatiale (positionnement, antérieur)** : Moran (1950) ; Bavaud (autocorrélation
  multivariée) ; Dray, Legendre & Peres-Neto (2006, *Moran Eigenvector Maps*) ; Dray, Saïd & Débias
  (2008, *MULTISPATI*). *(Cités comme antérieur : équivalence linéaire RV/autocorrélation déjà connue.)*
- **Statistique de forme — décomposition forme/taille (Prop. 7)** : Kendall (1984, shape space) ;
  Dryden & Mardia — *Statistical Shape Analysis* (2016). *(Lien forme=RV / taille=volume, signal JMVA.)*

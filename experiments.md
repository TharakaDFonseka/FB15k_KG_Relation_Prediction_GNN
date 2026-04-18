# Experiments

## Run-1

Launched via `run_rgcn_fb15k.slurm`: `python -u code/train.py --settings settings/gcn_basis.exp --dataset FB15k`.

| Part | Setting | Meaning |
|------|---------|--------|
| Encoder | `Name=gcn_basis` | R-GCN with basis decomposition (5 bases, 2 layers, 500-dim, dropout 0.8, etc.) |
| Decoder | `Name=bilinear-diag` | Diagonal bilinear scorer (DistMult-style; see equations below). |

### Encoder (`gcn_basis`) and “5 bases”

Settings (`settings/gcn_basis.exp`): `NumberOfBasisFunctions=5`, `NumberOfLayers=2`, `InternalEncoderDimension=500`, `DropoutKeepProbability=0.8`, `UseInputTransform=Yes`, `UseOutputTransform=No`. Implementation: `code/encoders/message_gcns/gcn_basis.py` (`BasisGcn`).

A full R-GCN could learn a **separate** weight matrix for every relation $r$. **Basis decomposition** instead shares a small set of **basis transforms** and learns, for each relation, **mixing coefficients** over those bases. Here the number of bases is $B = 5$ (`NumberOfBasisFunctions`).

Let $b \in \{1,\ldots,B\}$ index the bases. Conceptually, the relation-specific linear map used along an edge of type $r$ is a **linear combination** of $B$ shared maps rather than one unconstrained matrix per $r$:

$$
\mathbf{W}_r \;\approx\; \sum_{b=1}^{B} c_{r,b}\, \mathbf{V}_b
$$

The implementation keeps **separate** coefficient rows for forward- and backward-directed message passes (`C_forward`, `C_backward` in code), so for each direction,

$$
\mathbf{W}_r^{\rightarrow} = \sum_{b=1}^{B} c_{r,b}^{\rightarrow}\, \mathbf{V}_b^{\rightarrow}, \qquad
\mathbf{W}_r^{\leftarrow} = \sum_{b=1}^{B} c_{r,b}^{\leftarrow}\, \mathbf{V}_b^{\leftarrow}.
$$

Messages aggregate terms from the basis-transformed features; per edge type $r$, those terms are **weighted by** the corresponding $B$-vector $c_r$ (embedding lookup on relation id), then **summed over** $b=1,\ldots,B$—so **“5 bases”** means **five shared building blocks** that all relations reuse, with **five learned scalars per relation (per direction)** controlling the mix. This reduces parameters versus a full per-relation matrix at the cost of a fixed rank / shared structure.

### Decoder (`bilinear-diag`) equations

The decoder (`code/decoders/bilinear_diag.py`) maps each triple $(s, r, o)$ to embeddings $\mathbf{h}_s, \mathbf{r}_r, \mathbf{h}_o \in \mathbb{R}^d$ (from the encoder’s subject, relation, and object code matrices) and uses a **diagonal** interaction: score is the dot product after elementwise scaling by $\mathbf{r}_r$.

**Energy (logit) for a triple**

$$
f(s,r,o) = \sum_{i=1}^{d} [\mathbf{h}_s]_i \, [\mathbf{r}_r]_i \, [\mathbf{h}_o]_i
$$

Equivalently, $f(s,r,o) = (\mathbf{h}_s \odot \mathbf{r}_r)^\top \mathbf{h}_o$, where $\odot$ is elementwise product—this matches **DistMult** when each relation is a diagonal matrix $\mathrm{diag}(\mathbf{r}_r)$.

**Training loss**

Labels $y \in \{0,1\}$ are compared to the energy with TensorFlow’s `weighted_cross_entropy_with_logits` (binary classification loss on each sampled edge).

**Prediction**

$$
\hat{p}(s,r,o) = \sigma\!\bigl(f(s,r,o)\bigr)
$$

**Regularization** (decoder term; scaled by `RegularizationParameter` from settings)

$$
\lambda \left( \mathbb{E}\|\mathbf{h}_s\|^2 + \mathbb{E}\|\mathbf{r}_r\|^2 + \mathbb{E}\|\mathbf{h}_o\|^2 \right)
$$

(batch means over the minibatch in code).

## Run-2

Run-2 trains **basis + ComplEx**: **`gcn_basis`** encoder (**5** bases, same family as Run-1) with the **`complex`** decoder. This is **`settings/gcn_basis_complex.exp`** — **not** `complex.exp`, which pairs the ComplEx decoder with a shallow **`embedding`** encoder only.

| Part | Setting | Meaning |
|------|---------|--------|
| Encoder | `Name=gcn_basis`, `NumberOfBasisFunctions=5`, … | Same R-GCN basis stack as Run-1 (`BasisGcn`). |
| Decoder | `Name=complex` | ComplEx-style scoring (`code/decoders/complex.py`). |

**Launch (Slurm):** `run_gcn_basis_complex.slurm` — same resource/layout idea as `run_rgcn_fb15k.slurm`, but `cd` to this repo and:

```text
python -u code/train.py --settings settings/gcn_basis_complex.exp --dataset FB15k
```

**Slurm job ID:** __________________ (fill in after `sbatch`)

### Other `.exp` recipes (reference)

Each file is one **full training recipe** (encoder + decoder + dimensions + optimizer + batch settings). Changing file changes **how entity/relation vectors are produced** and/or **how a triple is scored**.

| File | Encoder | Decoder | In one sentence |
|------|---------|---------|-----------------|
| **`gcn_basis_complex.exp`** | `gcn_basis` → **`BasisGcn`** (**5** bases) | `complex` | **This Run-2 recipe** — R-GCN basis encoder + ComplEx head. |
| **`gcn_basis.exp`** | `gcn_basis` → **`BasisGcn`** (**5** bases) | `bilinear-diag` | Same encoder as Run-1 / basis+ComplEx, DistMult-style score. |
| **`gcn_block.exp`** | `gcn_basis` + **`ConcatGcn`** (**100** bases) | `bilinear-diag` | Larger / concat basis encoder + DistMult-style score. |
| **`distmult.exp`** | `embedding` (no GCN) | `bilinear-diag` | Shallow embeddings + DistMult. |
| **`complex.exp`** | `embedding` (no GCN) | `complex` | Shallow embeddings + ComplEx — **decoder only** matches Run-2; **encoder differs** from Run-2. |

So: **`gcn_basis` vs `gcn_block`** changes the **graph encoder**. **`distmult` vs `complex`** swaps the **decoder** with a fixed **embedding** encoder. **Run-2** fixes **basis encoder + ComplEx decoder** in one file (`gcn_basis_complex.exp`).

### The Complex decoder (what it does and where it lives)

**Name:** In settings, `Name=complex`. **Code:** `code/decoders/complex.py`, class `Complex`. **Not** a “distance model”: the name refers to **complex numbers** (real + imaginary parts), not “distance” in space.

**Idea in words:** For each triple $(s,r,o)$, the encoder still outputs three vectors—one for subject, one for relation, one for object. The Complex decoder **splits each of those vectors in half**: the first half is treated as the **real part**, the second half as the **imaginary part** of a complex embedding (per dimension). The score is then a **single real number** built from **four** three-way products (one per “complex interaction pattern”). That is the standard **ComplEx**-style scoring function: it can represent **asymmetric** patterns (e.g. relation vs inverse) more easily than a single real triple product.

Let $d$ be `CodeDimension` (must be **even**). For each entity/relation vector, write real and imaginary parts in $\mathbb{R}^{d/2}$ as $\mathbf{e}_s^{\Re}, \mathbf{e}_s^{\Im}$, $\mathbf{r}^{\Re}, \mathbf{r}^{\Im}$, $\mathbf{e}_o^{\Re}, \mathbf{e}_o^{\Im}$. The **energy (logit)** is

$$
f(s,r,o) = \sum_{k=1}^{d/2} \Bigl(
[\mathbf{e}_s^{\Re}]_k [\mathbf{r}^{\Re}]_k [\mathbf{e}_o^{\Re}]_k
+ [\mathbf{e}_s^{\Im}]_k [\mathbf{r}^{\Re}]_k [\mathbf{e}_o^{\Im}]_k
+ [\mathbf{e}_s^{\Re}]_k [\mathbf{r}^{\Im}]_k [\mathbf{e}_o^{\Im}]_k
- [\mathbf{e}_s^{\Im}]_k [\mathbf{r}^{\Im}]_k [\mathbf{e}_o^{\Re}]_k
\Bigr).
$$

This is the **real part** of the usual complex trilinear score $\Re\!\sum_k e_s^{(k)} r^{(k)} \overline{e_o^{(k)}}$ with complex units. **Predictions** still use $\hat{p}(s,r,o)=\sigma(f(s,r,o))$, and training uses the same **weighted cross-entropy on logits** pattern as the other decoders.

**Contrast with `bilinear-diag`:** There you multiply **three real vectors** per dimension and sum: $f = \sum_k h_s^k r^k h_o^k$. Here you still sum over dimensions, but each dimension uses **four** terms mixing real and imaginary parts—so it is **not** the same as DistMult unless you collapse to the purely real case.

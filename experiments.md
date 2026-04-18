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

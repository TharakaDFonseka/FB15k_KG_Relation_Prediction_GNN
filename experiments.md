# Experiments

## run1

Launched via `run_rgcn_fb15k.slurm`: `python -u code/train.py --settings settings/gcn_basis.exp --dataset FB15k`.

| Part | Setting | Meaning |
|------|---------|--------|
| Encoder | `Name=gcn_basis` | R-GCN with basis decomposition (5 bases, 2 layers, 500-dim, dropout 0.8, etc.) |
| Decoder | `Name=bilinear-diag` | Diagonal bilinear scorer (DistMult-style; see equations below). |

### Decoder (`bilinear-diag`) equations

The decoder (`code/decoders/bilinear_diag.py`) maps each triple \((s, r, o)\) to embeddings \(\mathbf{h}_s, \mathbf{r}_r, \mathbf{h}_o \in \mathbb{R}^d\) (from the encoder’s subject, relation, and object code matrices) and uses a **diagonal** interaction: score is the dot product after elementwise scaling by \(\mathbf{r}_r\).

**Energy (logit) for a triple**

\[
f(s,r,o) = \sum_{i=1}^{d} [\mathbf{h}_s]_i \, [\mathbf{r}_r]_i \, [\mathbf{h}_o]_i
\]

Equivalently, \(f(s,r,o) = (\mathbf{h}_s \odot \mathbf{r}_r)^\top \mathbf{h}_o\), where \(\odot\) is elementwise product—this matches **DistMult** when each relation is a diagonal matrix \(\mathrm{diag}(\mathbf{r}_r)\).

**Training loss**

Labels \(y \in \{0,1\}\) are compared to the energy with TensorFlow’s `weighted_cross_entropy_with_logits` (binary classification loss on each sampled edge).

**Prediction**

\[
\hat{p}(s,r,o) = \sigma\!\bigl(f(s,r,o)\bigr)
\]

**Regularization** (decoder term; scaled by `RegularizationParameter` from settings)

\[
\lambda \left( \mathbb{E}\|\mathbf{h}_s\|^2 + \mathbb{E}\|\mathbf{r}_r\|^2 + \mathbb{E}\|\mathbf{h}_o\|^2 \right)
\]

(batch means over the minibatch in code).

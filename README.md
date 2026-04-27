# FB15k_KG_Relation_Prediction_GNN

> **Note.** This project is a reimplementation of the original R-GCN work on relational link prediction, based on [*Modeling Relational Data with Graph Convolutional Networks*](https://arxiv.org/abs/1703.06103). We extend the encoder toward a **complex-valued** formulation (see code and `experiments.md` for details).

---
# Graph Convolutional Networks for Relational Link Prediction

This repository contains a TensorFlow implementation of Relational Graph Convolutional Networks (R-GCN), as well as experiments on relational link prediction. The description of the model and the results can be found in our paper:

[Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103). Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling (ArXiv 2017)

**Requirements**

* TensorFlow (1.4)

**Running demo**

We provide a bash script to run a demo of our code. In the folder *settings*, a collection of configuration files can be found. The block diagonal model used in our paper is represented through the configuration file *settings/gcn_block.exp*. To run a given experiment, execute our bash script as follows:

```
bash run-train.sh \[configuration\]
```

We advise that training can take up to several hours and require a significant amount of memory.

## Reimplementation Results (Current Status)

Using a `complex` decoder, we evaluated two encoder variants:

- `Ours_1`: 5-basis R-GCN
- `Ours_2`: block-diagonal R-GCN (`5x5` blocks, 100 blocks total)

### Full FB15k

| Model | MRR (raw) | MRR (filtered) | H@1 | H@3 | H@10 |
|------|-----------|----------------|-----|-----|------|
| `Ours_1` (5-basis) | 0.155 | 0.315 | 0.233 | 0.397 | 0.588 |
| `Ours_2` (block-diagonal) | 0.228 | 0.437 | 0.327 | 0.479 | 0.653 |

### FB15k-237

| Model | MRR (raw) | MRR (filtered) | H@1 | H@3 | H@10 |
|------|-----------|----------------|-----|-----|------|
| `Ours_1` (5-basis) | 0.099 | 0.153 | 0.087 | 0.164 | 0.285 |
| `Ours_2` (block-diagonal) | 0.151 | 0.242 | 0.153 | 0.258 | 0.413 |

### Technical interpretation

At this stage, our reproduced numbers do not outperform the reported values in the original R-GCN results. However, the trend is still promising: with the same `complex` decoder, the block-diagonal encoder consistently outperforms the 5-basis encoder on both datasets.

One likely reason is that the current settings (`5` bases for basis decomposition and `5x5` block structure with `100` blocks for block-diagonal) are not yet an optimal hyperparameter regime. In principle, the block-diagonal variant can benefit from reduced cross-relation weight sharing pressure compared to the low-basis setting, which may explain its stronger empirical performance here. A broader hyperparameter sweep is still needed before drawing final conclusions.

**Citation**

Please cite our paper if you use this code in your own work:

```
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and {van den Berg}, Rianne and Titov, Ivan and Welling, Max},
  booktitle={The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3--7, 2018, Proceedings 15},
  pages={593--607},
  year={2018},
  organization={Springer}
}
```

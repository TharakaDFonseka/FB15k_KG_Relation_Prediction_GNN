#!/bin/bash
#SBATCH --job-name=gcn_basis_complex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --partition=cpu
#SBATCH --output=/scratch/gpfs/JHA/tf8712/GNN/FB15k_KG_Relation_Prediction_GNN/logs/gcn_basis_complex_%j.out
#SBATCH --error=/scratch/gpfs/JHA/tf8712/GNN/FB15k_KG_Relation_Prediction_GNN/logs/gcn_basis_complex_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=tf8712@princeton.edu

set -eo pipefail

source /home/tf8712/.bashrc
conda activate rgcn_tf1

cd /scratch/gpfs/JHA/tf8712/GNN/FB15k_KG_Relation_Prediction_GNN
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

if [ ! -e FB15k ]; then
    ln -s data/FB15k FB15k
fi

python -u code/train.py --settings settings/gcn_basis_complex.exp --dataset FB15k

#!/bin/bash
#SBATCH --job-name=umls_map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --partition=cpu
#SBATCH --output=/scratch/gpfs/JHA/tf8712/GNN/RelationPrediction/logs/gcn_%j.out
#SBATCH --error=/scratch/gpfs/JHA/tf8712/GNN/RelationPrediction/logs/gcn_%j.err
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=tf8712@princeton.edu

set -eo pipefail

# initialize conda safely in batch shell
source /home/tf8712/.bashrc
conda activate rgcn_tf1

cd /scratch/gpfs/JHA/tf8712/GNN/RelationPrediction
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

# make sure dataset path exists
if [ ! -e FB15k ]; then
    ln -s data/FB15k FB15k
fi

python -u code/train.py --settings settings/gcn_basis.exp --dataset FB15k

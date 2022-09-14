#!/bin/bash


#SBATCH --job-name=ClusteringSemiSupervised
#SBATCH --nodes=1
#SBATCH --time=24:00:00 
#SBATCH --mem=64G
#SBATCH --partition=barton
#SBATCH --gres=gpu:1
#SBATCH --output=z_clustering_%j.txt 

. /etc/profile

module load lang/miniconda3/4.5.12
nvidia-smi

source activate tf-2.3

python3 -u BehaviorProfiling.py
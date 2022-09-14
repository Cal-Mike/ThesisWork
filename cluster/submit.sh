#!/bin/bash


#SBATCH --job-name=SemiSupervisedClustering
#SBATCH --nodes=1
#SBATCH --time=24:00:00 
#SBATCH --mem=64G
#SBATCH --output=z_clustering_%j.txt 

. /etc/profile

module load lang/miniconda3/4.5.12

source activate me4800

python3 -u dataExploration.py

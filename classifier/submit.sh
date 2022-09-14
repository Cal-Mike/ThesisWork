#!/bin/bash


#SBATCH --job-name=NewDataset
#SBATCH --nodes=1
#SBATCH --time=24:00:00 
#SBATCH --mem=64G
#SBATCH --output=READDATASET_%j.txt

. /etc/profile

module load lang/miniconda3/4.5.12

source activate tf-2.3

python3 -u dataExploration.py
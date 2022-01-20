#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=full2
#SBATCH --output=optim-full2.out
python3 main-full2.py
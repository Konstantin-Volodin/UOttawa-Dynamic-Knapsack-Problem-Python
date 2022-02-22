#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full2
#SBATCH --output=optim-smaller-full2.out
python3 main-smaller-full2.py
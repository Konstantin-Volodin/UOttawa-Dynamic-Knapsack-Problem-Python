#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full1
#SBATCH --output=optim-smaller-full1.out
python3 main-smaller-full1.py
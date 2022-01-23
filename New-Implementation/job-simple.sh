#!/bin/bash
#SBATCH --time=00-12
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=simple
#SBATCH --output=optim-simple.out
python3 main-simple.py
#!/bin/bash
#SBATCH --time=00-03
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=optimization
#SBATCH --output=optim-simple.out
python3 main-simple.py
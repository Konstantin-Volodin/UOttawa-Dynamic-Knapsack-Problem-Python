#!/bin/bash
#SBATCH --time=01-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=complex1
#SBATCH --output=optim-complex1.out
python3 main-complex1.py
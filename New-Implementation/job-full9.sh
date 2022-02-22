#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full9
#SBATCH --output=optim-full9.out
python3 main-full9.py
#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full7
#SBATCH --output=optim-full7.out
python3 main-full7.py
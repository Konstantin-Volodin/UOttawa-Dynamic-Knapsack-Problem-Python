#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full8
#SBATCH --output=optim-full8.out
python3 main-full8.py
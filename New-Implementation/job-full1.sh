#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=optimization
#SBATCH --output=optim-full1.out
python3 main-full1.py
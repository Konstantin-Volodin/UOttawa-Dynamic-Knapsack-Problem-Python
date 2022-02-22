#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full10
#SBATCH --output=optim-full10.out
python3 main-full10.py
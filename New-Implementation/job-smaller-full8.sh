#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full3
#SBATCH --output=optim-full3.out
python3 main-smaller-full8.py
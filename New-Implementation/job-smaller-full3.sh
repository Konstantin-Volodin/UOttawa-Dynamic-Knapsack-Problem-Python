#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full3
#SBATCH --output=optim-smaller-full3.out
python3 main-smaller-full3.py
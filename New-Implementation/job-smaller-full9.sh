#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full9
#SBATCH --output=optim-smaller-full9.out
python3 main-smaller-full9.py
#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full11
#SBATCH --output=optim-smaller-full11.out
python3 main-smaller-full11.py
#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full7
#SBATCH --output=optim-smaller-full7.out
python3 main-smaller-full7.py
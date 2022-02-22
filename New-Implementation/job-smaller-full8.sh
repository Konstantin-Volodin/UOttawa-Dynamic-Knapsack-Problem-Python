#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full8
#SBATCH --output=optim-smaller-full8.out
python3 main-smaller-full8.py
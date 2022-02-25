#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full14
#SBATCH --output=optim-smaller-full14.out
python3 main-smaller-full14.py
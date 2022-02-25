#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full12
#SBATCH --output=optim-smaller-full12.out
python3 main-smaller-full12.py
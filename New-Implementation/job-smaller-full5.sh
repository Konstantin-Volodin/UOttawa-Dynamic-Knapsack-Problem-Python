#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full5
#SBATCH --output=optim-smaller-full5.out
python3 main-smaller-full5.py
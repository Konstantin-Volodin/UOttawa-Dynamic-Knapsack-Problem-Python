#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full10
#SBATCH --output=optim-smaller-full10.out
python3 main-smaller-full10.py
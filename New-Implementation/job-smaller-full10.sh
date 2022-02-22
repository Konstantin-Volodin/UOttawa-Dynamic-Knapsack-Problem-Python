#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full5
#SBATCH --output=optim-full5.out
python3 main-smaller-full10.py
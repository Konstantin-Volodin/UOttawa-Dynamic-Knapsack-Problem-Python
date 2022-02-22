#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full1
#SBATCH --output=optim-full1.out
python3 main-smaller-full6.py
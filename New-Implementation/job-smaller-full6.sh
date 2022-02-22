#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full6
#SBATCH --output=optim-smaller-full6.out
python3 main-smaller-full6.py
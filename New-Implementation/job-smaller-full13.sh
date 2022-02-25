#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full13
#SBATCH --output=optim-smaller-full13.out
python3 main-smaller-full13.py
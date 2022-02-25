#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full15
#SBATCH --output=optim-smaller-full15.out
python3 main-smaller-full15.py
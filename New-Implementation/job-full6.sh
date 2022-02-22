#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full6
#SBATCH --output=optim-full6.out
python3 main-full6.py
#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=full3
#SBATCH --output=optim-full3.out
python3 main-full3.py
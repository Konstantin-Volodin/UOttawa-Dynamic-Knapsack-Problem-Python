#!/bin/bash
#SBATCH --time=01-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=optimization
#SBATCH --output=optim.out
python3 main.py
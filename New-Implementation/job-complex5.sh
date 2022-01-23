#!/bin/bash
#SBATCH --time=02-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=complex5
#SBATCH --output=optim-complex5.out
python3 main-complex5.py
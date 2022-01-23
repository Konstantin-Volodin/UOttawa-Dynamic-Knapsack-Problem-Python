#!/bin/bash
#SBATCH --time=02-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=complex2
#SBATCH --output=optim-complex2.out
python3 main-complex2.py
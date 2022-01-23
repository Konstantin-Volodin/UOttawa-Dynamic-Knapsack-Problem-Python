#!/bin/bash
#SBATCH --time=02-00
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=complex4
#SBATCH --output=optim-complex4.out
python3 main-complex4.py
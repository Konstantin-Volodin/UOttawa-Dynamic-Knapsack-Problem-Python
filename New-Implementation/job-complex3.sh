#!/bin/bash
#SBATCH --time=01-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=complex3
#SBATCH --output=optim-complex3.out
python3 main-complex3.py

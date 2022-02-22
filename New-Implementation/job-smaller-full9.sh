#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=full4
#SBATCH --output=optim-full4.out
python3 main-smaller-full9.py
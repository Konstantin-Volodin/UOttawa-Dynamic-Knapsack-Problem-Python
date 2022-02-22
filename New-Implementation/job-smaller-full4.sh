#!/bin/bash
#SBATCH --time=07-00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=smaller-full4
#SBATCH --output=optim-smaller-full4.out
python3 main-smaller-full4.py
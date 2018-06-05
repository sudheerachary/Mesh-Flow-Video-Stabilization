#!/bin/bash
#SBATCH -A research
#SBATCH -n 20
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

python parallel_mesh_flow.py

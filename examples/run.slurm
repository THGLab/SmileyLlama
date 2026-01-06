#!/bin/bash
#SBATCH -J llama_ft
#SBATCH -N 1
#SBATCH -p savio3_gpu
#SBATCH -A co_armada2
#SBATCH -q armada2_gpu3_normal
#SBATCH --gres=gpu:A40:4
#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output="slurm-%A.out"

source /global/home/users/ericwangyz/env/llama.env
echo "Host: ${HOSTNAME}"
echo "Acceletrate Path: $(which accelerate)"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"

python run.py


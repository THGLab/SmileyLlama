#!/bin/bash
#SBATCH -J SL
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --gres=gpu:A40:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output="slurm-%A.out"

source /global/home/users/ericwangyz/env/llama.env
echo "Host: ${HOSTNAME}"
echo "Acceletrate Path: $(which accelerate)"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"

sl sl_config_basic.yml


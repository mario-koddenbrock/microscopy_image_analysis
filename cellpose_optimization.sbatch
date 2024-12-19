#!/bin/bash

#SBATCH --job-name=cellpose
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Number of tasks (1 task that uses multiple GPUs)
#SBATCH --cpus-per-task=64         # Adjust CPUs per task based on the total available cores
#SBATCH --time=48:00:00            # Time limit (24 hours)
#SBATCH --mem=128G                  # Total memory for the job
#SBATCH --qos=normal
#SBATCH --output=cellpose_%j.log
#SBATCH --error=cellpose_%j.log


# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate microscopy_image_analysis

# Print the Python executable path for debugging
which python

# Run the evaluation script
python cellpose_optimization.py

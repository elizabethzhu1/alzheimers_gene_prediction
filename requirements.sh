#!/bin/bash

# Script to create a Conda environment with Python 3.12, pandas, and scikit-learn

# Name of the environment
ENV_NAME="my_env"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed or not added to PATH. Please install Conda and try again."
    exit 1
fi

# Create the environment
echo "Creating Conda environment: $ENV_NAME with Python 3.12..."
conda create -n $ENV_NAME python=3.12 -y

# Activate the environment
echo "Activating the Conda environment: $ENV_NAME..."
source activate $ENV_NAME

# Install packages
echo "Installing pandas and scikit-learn..."

pip install -U pandas
pip install -U scikit-learn
pip install -U matplotlib

# Confirm installation
echo "Environment $ENV_NAME created successfully with the following packages:"
conda list

# Exit
echo "To activate the environment, use: conda activate $ENV_NAME"
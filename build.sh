#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies required for OpenCV and scikit-image
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    build-essential \
    cmake

# Upgrade pip and install basic build tools
python -m pip install --upgrade pip
pip install setuptools wheel

# Install numpy first (OpenCV dependency)
pip install numpy==1.24.3

# Install the rest of the requirements
pip install -r requirements.txt 
#!/bin/bash
# RAPIDS Environment Setup Script
# This script sets up the necessary library paths for RAPIDS cuML, cuDF, and CuPy

export LD_LIBRARY_PATH="/home/jonahs23/.local/lib/python3.12/site-packages/libcuml/lib64:/home/jonahs23/.local/lib/python3.12/site-packages/libcudf/lib64:/home/jonahs23/.local/lib/python3.12/site-packages/libraft/lib64:/home/jonahs23/.local/lib/python3.12/site-packages/librmm/lib64:/home/jonahs23/.local/lib/python3.12/site-packages/libucxx/lib64:/opt/anaconda3/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

echo "RAPIDS environment configured!"
echo "You can now use cuML, cuDF, and CuPy in your Python scripts."

#!/bin/bash
# RAPIDS Environment Setup Script for ads_5984
# Source this script before starting Jupyter or running Python with RAPIDS

export LD_LIBRARY_PATH="/home/jonahs23/.local/lib/python3.10/site-packages/libcuml/lib64:/home/jonahs23/.local/lib/python3.10/site-packages/libcudf/lib64:/home/jonahs23/.local/lib/python3.10/site-packages/libraft/lib64:/home/jonahs23/.local/lib/python3.10/site-packages/librmm/lib64:/home/jonahs23/.local/lib/python3.10/site-packages/libucxx/lib64:/opt/anaconda3/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

echo "RAPIDS environment configured for ads_5984!"
echo "LD_LIBRARY_PATH has been set for cuML, cuDF, and CuPy."

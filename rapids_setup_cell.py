# Add this cell at the TOP of your Jupyter notebook (before importing cuML)
# This sets up the library paths needed for RAPIDS packages

import os

# Set LD_LIBRARY_PATH for RAPIDS GPU libraries
rapids_lib_paths = [
    "/home/jonahs23/.local/lib/python3.10/site-packages/libcuml/lib64",
    "/home/jonahs23/.local/lib/python3.10/site-packages/libcudf/lib64",
    "/home/jonahs23/.local/lib/python3.10/site-packages/libraft/lib64",
    "/home/jonahs23/.local/lib/python3.10/site-packages/librmm/lib64",
    "/home/jonahs23/.local/lib/python3.10/site-packages/libucxx/lib64",
    "/opt/anaconda3/lib/python3.12/site-packages/nvidia/nccl/lib"
]

existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_path = ':'.join(rapids_lib_paths)
if existing_ld_path:
    new_ld_path = f"{new_ld_path}:{existing_ld_path}"

os.environ['LD_LIBRARY_PATH'] = new_ld_path

print("✓ RAPIDS library paths configured successfully!")
print("You can now import cuml, cudf, and cupy")

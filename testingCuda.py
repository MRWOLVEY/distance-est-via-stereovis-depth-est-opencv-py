import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# import sys
# import os
# print("Python Executable:", sys.executable)
# print("Python Version:", sys.version)
# print("Environment Variables:", os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX'))
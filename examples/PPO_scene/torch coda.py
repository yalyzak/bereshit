import torch

# Define the device dynamically to use CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print the selected device
print('Using device:', device)

# Optional: Print additional information if using a CUDA device
if device.type == 'cuda':
    print('GPU Device name:', torch.cuda.get_device_name(0))
    print('Memory Usage Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')

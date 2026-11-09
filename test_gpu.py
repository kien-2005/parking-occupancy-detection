import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    x = torch.rand(1000, 1000, device="cuda")
    y = x @ x

    print("GPU test successful!")
else:
    print("GPU chưa hoạt động!")
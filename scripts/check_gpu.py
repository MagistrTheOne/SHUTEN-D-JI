"""Quick GPU check script."""
import torch

print("=== GPU CHECK ===")
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1e9
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)} - {mem_gb:.1f}GB")
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("=== OK ===")

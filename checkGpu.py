import torch

# 1. Check if the GPU is visible
if torch.cuda.is_available():
    print(f"✅ SUCCESS: Found GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Check how much VRAM you have (16GB for 5060 Ti?)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM Available: {vram:.2f} GB")
    
    # 3. Run a quick test calculation on the GPU
    x = torch.rand(5, 5).cuda()
    print("   Test Tensor successfully created on GPU!")
else:
    print("❌ ERROR: Running on CPU. Drivers or PyTorch not installed correctly.")
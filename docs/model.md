from safetensors.torch import load_file

f = load_file("D:/models/unet/flux1-dev.safetensors")

keys = [k for k in f]

for k in keys:
    print(f"{k} {f.pop(k).shape}")
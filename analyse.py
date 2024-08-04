from safetensors.torch import load_file
from scipy.stats import spearmanr
import torch

all = list(load_file(f"../../{i}.internals.safetensors") for i in range(4))
keys = list(k for k in all[0])

for k in keys:
    print(f"\n{k}")
    for i in range(4):
        for j in range(i+1,4):
            r = spearmanr(all[i][k].to(torch.float),all[j][k].to(torch.float))
            print(f"{i} {j} {r.statistic}")
    
import torch
from tqdm import trange
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
from .utils import is_double, shared

def new_layer(s, n) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    if is_double(n):
        return DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    else:
        return SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)

def load_single_layer(layer_number:int, remove_from_sd=True, dry_run=False) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    layer_sd = shared.layer_sd(layer_number)
    if remove_from_sd: shared.drop_layer(layer_number)
    if dry_run: return torch.nn.Linear(in_features=1, out_features=1)
    layer = new_layer(None, layer_number)
    layer.load_state_dict(layer_sd)
    return layer

def load_layer_stack(dry_run=False):
    print("Loading model")
    layer_stack = torch.nn.Sequential( *[load_single_layer(layer_number=x, dry_run=dry_run) for x in trange(shared.last_layer+1)] )
    return layer_stack
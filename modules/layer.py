from .utils import shared, is_double, log
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import torch

def load_single_layer(layer_number:int) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    log(f"Loading layer {layer_number}")
    if is_double(layer_number):
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
        prefix = f"double_blocks.{layer_number}."
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)
        prefix = f"single_blocks.{layer_number - shared.last_double_layer - 1}."

    layer_sd = { k[len(prefix):]:shared.sd[k] for k in shared.sd if k.startswith(prefix) }
    layer.load_state_dict(layer_sd)
    return layer
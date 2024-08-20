from .utils import shared, is_double, log, prefix
from .pruning import prune_layer
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import torch

def load_single_layer(layer_number:int) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    log(f"Loading layer {layer_number}")
    layer_prefix = prefix(layer_number)
    layer_sd = { k[len(layer_prefix):]:shared.sd[k] for k in shared.sd if k.startswith(layer_prefix) }
    if is_double(layer_number):
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)
    prune_layer(layer, layer_number, 12288-int(layer_sd['txt_mlp.0.bias'].shape[0]), "txt", None)
    prune_layer(layer, layer_number, 12288-int(layer_sd['img_mlp.0.bias'].shape[0]), "img", None)
    layer.load_state_dict(layer_sd)
    return layer
from .utils import shared, is_double, log, prefix
from .pruning import prune_layer
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import torch

def load_single_layer(layer_number:int, remove_from_sd=False) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    log(f"Loading layer {layer_number}")
    layer_sd = shared.layer_sd(layer_number)
    if remove_from_sd: shared.drop_layer(layer_number)
    if is_double(layer_number):
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
        prune_layer(layer, layer_number, int(layer.txt_mlp[0].bias.shape[0] - layer_sd['txt_mlp.0.bias'].shape[0]), "txt")
        prune_layer(layer, layer_number, int(layer.img_mlp[0].bias.shape[0] - layer_sd['img_mlp.0.bias'].shape[0]), "img")
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)
        prune_layer(layer, layer_number, int(layer.linear1.bias.shape[0] - layer_sd['linear1.bias'].shape[0]), 'x')
    layer.load_state_dict(layer_sd)
    return layer
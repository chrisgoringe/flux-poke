import yaml
import torch
from .utils import range_from_string, log, shared
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union

class CastLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, to):
        super().__init__()
        self.linear  = linear.to(to)

    def forward(self, x):
        with torch.autocast("cuda"): return self.linear(x)

def cast_layer(layer:Union[DoubleStreamBlock, SingleStreamBlock], block_pattern:str, cast_to, callback:callable = None):
    for blockname, block in layer.named_children():
        if block_pattern=='_' or block_pattern in blockname:
            if isinstance(block, torch.nn.Linear):
                setattr(layer, blockname, CastLinear(block,cast_to))
                if callback: callback(blockname)
            else:
                for name, module in block.named_children():
                    if isinstance(module, torch.nn.Linear):
                        setattr(block, name, CastLinear(module,cast_to))
                        if callback: callback(f"{blockname}.{name}")

def cast_model(model, cast_map_filepath, layer_index=0, thickness=1):
    with open(cast_map_filepath, 'r') as f: map = yaml.safe_load(f)
    for mod in map['mods']:
        if (cast:=mod.get('cast', 'default')) != 'none':
            cast_to   = getattr(torch, (cast if cast!="default" else map['default']))
            block_pattern = mod.get('blocks', '_')
            for global_layer_index in range_from_string(mod.get('layers',None)):
                model_layer_index = global_layer_index - layer_index
                if model_layer_index>=0 and model_layer_index<thickness:
                    layer = model[model_layer_index]
                    def record(linear_name): shared.layer_stats[global_layer_index][linear_name] = f"{cast_to}"
                    cast_layer(layer, block_pattern, cast_to, record)
                    
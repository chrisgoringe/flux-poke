import torch
from .utils import int_list_from_string, log, shared
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

from typing import Union
import bitsandbytes.nn as bnb
from gguf import GGMLQuantizationType, quants
from .city_gguf.ops import GGMLOps

def quantise_tensor(t:torch.Tensor, gtype:GGMLQuantizationType):
    if t.dtype not in (torch.float16, torch.float32):
        t = t.to(torch.float32)
    t = quants.quantize(t.squeeze().numpy(), gtype)

class CastLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, to):
        super().__init__()
        if hasattr(to, '__call__'):  
            self.linear = to(linear.in_features, linear.out_features, linear.bias is not None, device=linear.weight.device)
            self.linear.load_state_dict(linear.state_dict())
        elif isinstance(to, GGMLQuantizationType):
            self.linear = GGMLOps.Linear(linear.in_features, linear.out_features, linear.bias is not None, device=linear.weight.device)
            sd = linear.state_dict()
            self.linear.load_state_dict({k:quantise_tensor(sd[k], to)} for k in sd)
        else:
            self.linear = linear.to(to)

    def forward(self, x:torch.Tensor):
        return self.linear(x)

def cast_layer(layer:Union[DoubleStreamBlock, SingleStreamBlock], cast_to, block_constraint:str = None, callbacks:list[callable] = []):
    def recursive_cast(parent_module:torch.nn.Module, parent_name:str, child_module:torch.nn.Module, child_name:str):
        child_fullname = ".".join((parent_name,child_name))
        if isinstance(child_module, torch.nn.Linear):
            if block_constraint is None or block_constraint in child_fullname:
                setattr(parent_module, child_name, CastLinear(child_module,cast_to))
                for callback in callbacks: callback(child_fullname, cast_to)
        else:
            for grandchild_name, grandchild_module in child_module.named_children():
                recursive_cast(child_module, child_fullname, grandchild_module, grandchild_name)

    for child_name, child_module in layer.named_children():
        recursive_cast(layer, "", child_module, child_name)

def cast_layer_stack(layer_stack, cast_config, stack_starts_at_layer, default_cast, verbose=False, callbacks=[]):
    for mod in cast_config.get('casts',None) or []:
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint=None
        if (cast:=mod.get('castto', 'default')) != 'none' and block_constraint != 'none':
            if cast == "default": cast = default_cast

            cast_to = None
            for source in [bnb, torch, GGMLQuantizationType]:
                if hasattr(source, cast):
                    cast_to = getattr(source, cast)
                    break
            if cast_to is None: raise NotImplementedError(f"Type {cast} not known")

            for global_layer_index in int_list_from_string(mod.get('layers',None)):
                model_layer_index = global_layer_index - stack_starts_at_layer
                if model_layer_index>=0 and model_layer_index<len(layer_stack):
                    layer = layer_stack[model_layer_index]
                    def record(linear_name, _): 
                        if verbose: print(f"Cast {global_layer_index}{linear_name} to {cast_to}")
                        shared.layer_stats[global_layer_index][linear_name] = f"{cast_to}"
                    cast_layer(layer=layer, cast_to=cast_to, block_constraint=block_constraint, callbacks=callbacks + [record,])

import torch
import asyncio, time
from .utils import layer_iteratable_from_string, log, shared
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

from typing import Union
import bitsandbytes.nn as bnb
from warnings import warn

from .gguf_py.gguf import GGMLQuantizationType, quants
from .city_gguf.dequant import dequantize
from .async_prepare import async_run_prepares

class TensorPlus(object):
    def __init__(self, data, gtype:GGMLQuantizationType, oshape:torch.Size, **kwargs):
        self._tensor = torch.as_tensor(data, **kwargs)
        self.gtype = gtype
        self.oshape = oshape

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        gtypes = tuple(a.gtype for a in args if hasattr(a, 'gtype'))
        oshapes = tuple(a.oshape for a in args if hasattr(a, 'oshape'))
        args = [getattr(a, '_tensor', a) for a in args]
        ret = func(*args, **kwargs)
        return TensorPlus(ret, gtype=gtypes[0], oshape=oshapes[0])

def quantise_tensor(t:torch.Tensor, gtype:GGMLQuantizationType) -> TensorPlus:
    if t is None: return None
    t = t.to(torch.float32)
    return TensorPlus(quants.quantize(t.squeeze().numpy(), gtype), gtype=gtype, oshape=t.shape)

def dequantize_tensor(tensor:TensorPlus, dtype, device):
    out = dequantize(tensor._tensor, tensor.gtype, tensor.oshape, dtype=dtype)
    return out.to(dtype).to(device)

class DequantingLinear(torch.nn.Module):
# Based on https://github.com/city96/ComfyUI-GGUF (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
    def __init__(self, sd:dict[str,torch.Tensor], qtype:GGMLQuantizationType, dtype=None, device=None):
        super().__init__()
        self.weight = quantise_tensor(sd['weight'], qtype)
        self.bias   = quantise_tensor(sd.get('bias',None), qtype) 
        self.qtype  = qtype
        self.prepared  = None
        self.preparing = False
        self.dtype = dtype
        self.device = device

    def move_patch_to_cuda(self, item, device):
        if isinstance(item, torch.Tensor):
            return item.to(device, non_blocking=True)
        elif isinstance(item, tuple):
            return tuple(self.move_patch_to_cuda(x, device) for x in item)
        elif isinstance(item, list):
            return [self.move_patch_to_cuda(x, device) for x in item]
        else:
            return item

    def get_weight(self, tensor:TensorPlus, dtype, device):
        # consolidate and load patches to GPU in async
        patch_list = []

        for function, patches, _ in getattr(tensor, "patches", []):
            patch_list += self.move_patch_to_cuda(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, device)

        # apply patches
        if patch_list: weight = function(patch_list, weight, "dequant.name.unknown")
        return weight

    def get_weight_and_bias(self, dtype, device):
        weight = self.get_weight(self.weight, dtype, device)
        bias   = self.get_weight(self.bias,   dtype, device) if self.bias is not None else None
        return (weight, bias)
    
    def prepare(self):
        if self.dtype and self.device:
            self.preparing = True
            self.prepared = self.get_weight_and_bias(dtype=self.dtype, device=self.device)
            self.preparing = False
        else:
            warn("Can't prepare without dtype and device hints")

    def forward(self, x:torch.Tensor):
        while (self.preparing): asyncio.sleep(0)

        if self.prepared:
            weight, bias = self.prepared
            self.prepared = None
        else:
            weight, bias = self.get_weight_and_bias(dtype=x.dtype, device=x.device)
        x = torch.nn.functional.linear(x, weight, bias)
        del weight, bias
        return x
    
class CastLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, to, autocast=False):
        super().__init__()
        self.description = str(to)
        self.autocast = autocast
        if hasattr(to, '__call__'):  
            self.linear = to(linear.in_features, linear.out_features, linear.bias is not None, device=linear.weight.device)
            self.linear.load_state_dict(linear.state_dict())
        elif isinstance(to, GGMLQuantizationType):
            self.linear = DequantingLinear(linear.state_dict(), qtype=to, dtype=torch.bfloat16, device="cuda")
            self.description = to.name
        else:
            self.linear = linear.to(to)

    def forward(self, x:torch.Tensor):
        with torch.autocast(device_type="cuda", enabled=self.autocast):
            return self.linear(x)

def cast_layer(layer:Union[DoubleStreamBlock, SingleStreamBlock], cast_to, block_constraint:str = None, callbacks:list[callable] = [], initial_name="", autocast=False):
    def recursive_cast(parent_module:torch.nn.Module, parent_name:str, child_module:torch.nn.Module, child_name:str):
        child_fullname = ".".join((parent_name,child_name))
        if isinstance(child_module, torch.nn.Linear):
            if block_constraint is None or block_constraint in child_fullname:
                cast_value = CastLinear(child_module,cast_to,autocast)
                setattr(parent_module, child_name, cast_value)
                for callback in callbacks: callback(child_fullname, cast_value.description)
        else:
            for grandchild_name, grandchild_module in child_module.named_children():
                recursive_cast(child_module, child_fullname, grandchild_module, grandchild_name)

    for child_name, child_module in layer.named_children():
        recursive_cast(layer, initial_name, child_module, child_name)

def cast_layer_stack(layer_stack, cast_config, stack_starts_at_layer, default_cast, verbose=False, callbacks=[], autocast=False):
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

            for global_layer_index in layer_iteratable_from_string(mod.get('layers',None)):
                model_layer_index = global_layer_index - stack_starts_at_layer
                if model_layer_index>=0 and model_layer_index<len(layer_stack):
                    layer = layer_stack[model_layer_index]
                    def record(linear_name, cast_name): 
                        if verbose: print(f"Cast {linear_name} to {cast_name}")
                        shared.layer_stats[global_layer_index][linear_name] = f"{cast_name}"
                    cast_layer(layer=layer, 
                               cast_to=cast_to, 
                               block_constraint=block_constraint, 
                               callbacks=callbacks + [record,],
                               initial_name=f"{global_layer_index}",
                               autocast=autocast)
                    async_run_prepares(layer)


import torch
from .utils import layer_iteratable_from_string, shared
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

from typing import Union
import bitsandbytes.nn as bnb
from warnings import warn

from .gguf_py.gguf import GGMLQuantizationType, quants
from .city_gguf.dequant import dequantize
from functools import partial
from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader
import numpy as np

class QuantizedTensor(object):
    def __init__(self, data, gtype:GGMLQuantizationType, oshape:torch.Size, **kwargs):
        self._tensor = data if isinstance(data, torch.Tensor) or data is None else torch.as_tensor(data)  
        self.tensor_type = gtype
        self.tensor_shape = oshape
        self.patches = []
        self._dequanted = None

    def wrap(self, fn, *args, **kwargs):
        x = fn(*args, **kwargs)
        return QuantizedTensor(x, self.tensor_type, self.tensor_shape) if isinstance(x, torch.Tensor) else x
    
    def __getattr__(self, __name: str):
        a = getattr(self._tensor, __name)
        return partial(self.wrap, a) if hasattr(a,'__call__') else a
    
    def dequanted(self, dtype, device):
        if self._dequanted is None:
            self._dequanted = dequantize_tensor(self, dtype, device)
        return self._dequanted
    
    def purge(self):
        self._dequanted = None
    
    @classmethod
    def load_from_reader_tensor(cls, reader_tensor:ReaderTensor):
        return QuantizedTensor( data=reader_tensor.data, tensor_type=reader_tensor.tensor_type, tensor_shape=torch.Size(np.flip(list(reader_tensor.shape))))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        gtypes = tuple(a.tensor_type for a in args if hasattr(a, 'tensor_type'))
        oshapes = tuple(a.tensor_shape for a in args if hasattr(a, 'tensor_shape'))
        args = [getattr(a, '_tensor', a) for a in args]
        ret = func(*args, **kwargs)
        return QuantizedTensor(ret, gtype=gtypes[0], oshape=oshapes[0])

def quantise_tensor(t:torch.Tensor, gtype:GGMLQuantizationType) -> QuantizedTensor:
    if t is None: return None
    t = t.to(torch.float32)
    return QuantizedTensor(quants.quantize(t.squeeze().numpy(), gtype), gtype=gtype, oshape=t.shape)

def dequantize_tensor(tensor:QuantizedTensor, dtype, device):
    if tensor is None: return None
    out = dequantize(tensor._tensor, tensor.tensor_type, tensor.tensor_shape, dtype=dtype)
    return out.to(dtype).to(device)

class DequantingLinear(torch.nn.Module):
# Based on https://github.com/city96/ComfyUI-GGUF (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
    def __init__(self, sd:dict[str,torch.Tensor]=None, qtype:GGMLQuantizationType=None, reader_tensor_weight:ReaderTensor=None, reader_tensor_bias:ReaderTensor=None):
        super().__init__()
        if reader_tensor_weight is not None:
            assert sd is None
            self.weight = QuantizedTensor.load_from_reader_tensor(reader_tensor_weight)
            self.bias   = QuantizedTensor.load_from_reader_tensor(reader_tensor_bias) if reader_tensor_bias is not None else None
        else:
            self.weight = quantise_tensor(sd['weight'], qtype)
            self.bias   = quantise_tensor(sd.get('bias',None), qtype) 

    def _apply(self, fn):
        if self.weight is not None:
            try:
                self.weight = fn(self.weight)
            except TypeError:
                pass # why?
        if self.bias is not None:
            self.bias = fn(self.bias)
        super()._apply(fn)
        return self

    def move_patch_to_cuda(self, item, device):
        if isinstance(item, torch.Tensor):
            return item.to(device, non_blocking=True)
        elif isinstance(item, tuple):
            return tuple(self.move_patch_to_cuda(x, device) for x in item)
        elif isinstance(item, list):
            return [self.move_patch_to_cuda(x, device) for x in item]
        else:
            return item

    def get_weight(self, tensor:QuantizedTensor, dtype, device):
        # consolidate and load patches to GPU in async
        patch_list = []

        for function, patches, _ in getattr(tensor, "patches", []):
            patch_list += self.move_patch_to_cuda(patches, device)

        # dequantize tensor while patches load
        weight = tensor.dequanted(dtype, device)

        # apply patches
        if patch_list: weight = function(patch_list, weight, "dequant.name.unknown")
        return weight

    def get_weight_and_bias(self, dtype, device):
        weight = self.get_weight(self.weight, dtype, device)
        bias   = self.get_weight(self.bias,   dtype, device) if self.bias is not None else None
        return (weight, bias)
    
    def prepare(self, dtype, device):
        self.weight.dequanted(dtype, device)
        if self.bias is not None: self.bias.dequanted(dtype, device)

    def forward(self, x:torch.Tensor):
        weight, bias = self.get_weight_and_bias(dtype=x.dtype, device=x.device)
        x = torch.nn.functional.linear(x, weight, bias)
        del weight, bias
        self.weight.purge()
        if self.bias is not None: self.bias.purge()
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
            self.linear = DequantingLinear(linear.state_dict(), qtype=to)
            self.description = to.name
        else:
            self.linear = linear.to(to)

    def forward(self, x:torch.Tensor):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self.linear(x)
        else:
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


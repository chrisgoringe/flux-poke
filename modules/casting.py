import torch
from .utils import layer_iteratable_from_string, shared
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

from typing import Union
import bitsandbytes.nn as bnb
from warnings import warn

from gguf import GGMLQuantizationType, quants
from .city_gguf.dequant import dequantize

from gguf.gguf_reader import ReaderTensor
from gguf.quants import quantize
import numpy as np

from functools import partial

class QuantizedTensor():
    def __init__(self, data=None, tensor_type=None, tensor_shape=None, patches=[], data_is_unquantized_tensor=False, **kwargs):
        self.tensor_type:GGMLQuantizationType = tensor_type
        self.tensor_shape:torch.Size          = tensor_shape
        self.patches:list                     = patches.copy()
        self._tensor:torch.Tensor             = None
        self._set_data(data, data_is_unquantized_tensor)

    def dequantized(self, dtype, device=None) -> torch.Tensor:
        return dequantize(self._tensor, self.tensor_type, self.tensor_shape, dtype=dtype).to(dtype).to(device)

    @property
    def tensor_description(self):
        try:
            return torch.tensor([int(self.tensor_type),] + [int(x) for x in self.tensor_shape], device="cpu")
        except:
            raise Exception()
    
    @classmethod
    def load_from_description(cls, description, tnsr):
        return QuantizedTensor( data=tnsr, tensor_type=int(description[0]), tensor_shape=torch.Size(description[1:]))
    
    @classmethod
    def load_from_reader_tensor(cls, reader_tensor:ReaderTensor):
        return QuantizedTensor( data=reader_tensor.data, tensor_type=reader_tensor.tensor_type, tensor_shape=torch.Size(np.flip(list(reader_tensor.shape))))

    @classmethod
    def from_unquantized_tensor(cls, tnsr:torch.Tensor, tensor_type:GGMLQuantizationType):
        return QuantizedTensor( tnsr, tensor_shape=tnsr.shape, tensor_type=tensor_type, data_is_unquantized_tensor=True )
    
    def _set_data(self, data, data_is_unquantized_tensor=False):
        if data_is_unquantized_tensor:
            assert isinstance(data, torch.Tensor)
            try:              data = quantize(data.numpy(), qtype=self.tensor_type)
            except TypeError: data = quantize(data.to(torch.float).numpy(), qtype=self.tensor_type)
        self._tensor = data if isinstance(data, torch.Tensor) or data is None else torch.as_tensor(data)  

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for a in args:
            if isinstance(a, QuantizedTensor):
                return_qt = QuantizedTensor(None, a.tensor_type, a.tensor_shape, a.patches)
                break

        args = [getattr(a, '_tensor', a) for a in args]
        return_qt._set_data( func(*args, **kwargs) )
        return return_qt
    
    def wrap(self, fn, *args, **kwargs):
        x = fn(*args, **kwargs)
        return QuantizedTensor(x, self.tensor_type, self.tensor_shape, self.patches) if isinstance(x, torch.Tensor) else x
    
    def __getattr__(self, __name: str):
        a = getattr(self._tensor, __name)
        return partial(self.wrap, a) if hasattr(a,'__call__') else a

def quantise_tensor(t:torch.Tensor, gtype:GGMLQuantizationType) -> QuantizedTensor:
    if t is None: return None
    t = t.to(torch.float32)
    return QuantizedTensor(quants.quantize(t.squeeze().numpy(), gtype), tensor_type=gtype, tensor_shape=t.shape)

def dequantize_tensor(tensor:QuantizedTensor, dtype, device):
    if tensor is None: return None
    out = dequantize(tensor._tensor, tensor.tensor_type, tensor.tensor_shape, dtype=dtype)
    return out.to(dtype).to(device)

class DequantingLinear(torch.nn.Module):
    def __init__(self, sd:dict[str,torch.Tensor]=None, qtype:GGMLQuantizationType=None, reader_tensor_weight:ReaderTensor=None, reader_tensor_bias:ReaderTensor=None):
        super().__init__()
        if reader_tensor_weight is not None:
            assert sd is None
            assert isinstance(reader_tensor_weight, ReaderTensor)
            self.weight = QuantizedTensor.load_from_reader_tensor(reader_tensor_weight)
            self.bias   = QuantizedTensor.load_from_reader_tensor(reader_tensor_bias) if reader_tensor_bias is not None else None
        else:
            self.weight = quantise_tensor(sd['weight'], qtype)
            self.bias   = quantise_tensor(sd.get('bias',None), qtype) 

    @classmethod
    def from_reader_tensors(cls, weight_and_bias:tuple[ReaderTensor,Union[ReaderTensor,None]]):
        w, b = weight_and_bias
        return DequantingLinear(reader_tensor_weight=w, reader_tensor_bias=b)

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

    def move_things(self, thing, device):
        if isinstance(thing, torch.Tensor): return thing.to(device, non_blocking=True)
        elif isinstance(thing, tuple):      return tuple(self.move_things(x, device) for x in thing)
        elif isinstance(thing, list):       return [self.move_things(x, device) for x in thing]
        else:                               return thing

    def get_weight(self, tensor:QuantizedTensor, dtype, device):
        patch_list = []
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += self.move_things(patches, device)

        weight = tensor.dequantized(dtype, device)

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


import torch
from .utils import layer_iteratable_from_string
from .casting import DequantingLinear

from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader

class PatchingMap:
    def __init__(self):
        self.map_linear_to_parent:dict[str,torch.nn.Module] = {}
        self.map_linear_to_tensors:dict[str,dict[str,ReaderTensor]] = {}

    def add_linear(self, fullname:str, parent:torch.nn.Module):
        self.map_linear_to_parent[fullname]  = parent
        self.map_linear_to_tensors[fullname] = {}

    def add_tensor(self, linear_name:str, tensor_name:str, tensor:ReaderTensor):
        self.map_linear_to_tensors[linear_name][tensor_name] = tensor

    def get_weight_and_bias(self, linear_name:str) -> tuple[ReaderTensor, ReaderTensor]:
        return ( self.map_linear_to_tensors[linear_name].get('weight', None), self.map_linear_to_tensors[linear_name].get('bias', None) )
    
    def get_parent(self, linear_name:str) -> torch.nn.Module:
        return self.map_linear_to_parent[linear_name]
    
    def get_childname(self, linear_name:str) -> str:
        return linear_name.split(".")[-1] 

    def __contains__(self, name):
        return name in self.map_linear_to_parent 
    
    def __iter__(self): 
        for linear in self.map_linear_to_parent: yield linear


def name_all_linears(layer_stack, layer_list, block_constraint) -> PatchingMap:
    linear_map = PatchingMap()

    def all_linears(name:str, module:torch.nn.Module):
        for n,m in module.named_children():
            nm = ".".join((name, n))
            if isinstance(m,torch.nn.Linear):
                if block_constraint is None or block_constraint in nm:
                    linear_map.add_linear(nm, module)
            else:
                all_linears(nm, m)

    for i, layer in enumerate(layer_stack):
        if i in layer_list:
            name = f"double_blocks.{i}" if i<=18 else f"single_blocks.{i-19}"
            all_linears(name, layer)

    return linear_map

def patch_layer_stack(layer_stack, patch_config, verbose):
    for mod in patch_config['patches']:
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint=None

        layers = list(layer_iteratable_from_string(mod['layers']))
        linear_map = name_all_linears(layer_stack, layers, block_constraint)

        reader = GGUFReader(mod['file'])
        tensor:ReaderTensor
        for tensor in reader.tensors:
            most, last = ".".join(tensor.name.split(".")[:-1]) , tensor.name.split(".")[-1] 
            if most in linear_map:
                linear_map.add_tensor(most, last, tensor)

        for linear in linear_map:
            if verbose: print(f"Patching {linear}")
            setattr(
                    linear_map.get_parent(linear), 
                    linear_map.get_childname(linear), 
                    DequantingLinear.from_reader_tensors( weight_and_bias=linear_map.get_weight_and_bias(linear) )
                    )


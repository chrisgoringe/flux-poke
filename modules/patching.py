import torch
from .utils import layer_iteratable_from_string
from .casting import QuantizedTensor

from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader


def name_all_linears(layer_stack, layer_list, block_constraint):
    name_linear_map:dict[str,torch.nn.Module] = {}

    def all_linears(name:str, module:torch.nn.Module):
        for n,m in module.named_children():
            nm = ".".join(name, n)
            if isinstance(m,torch.nn.Linear):
                if block_constraint is None or block_constraint in nm:
                    name_linear_map[nm] = m
            else:
                all_linears(nm, m)

    for i, layer in enumerate(layer_stack):
        if i in layer_list:
            name = f"double_blocks.{i}" if i<=18 else f"single_blocks.{i-19}"
            all_linears(name, layer)

    return name_linear_map

def patch_layer_stack(layer_stack, patch_config, verbose):
    for mod in patch_config['patches']:
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint=None

        layers = list(layer_iteratable_from_string(mod['layers']))
        linear_map = name_all_linears(layer_stack, layers, block_constraint)

        reader = GGUFReader(mod['file'])
        tensor:ReaderTensor
        for tensor in reader.tensors:
            most, last = ".".join(tensor.split(".")[:-1]) , tensor.split(".")[-1] 
            if most in linear_map:
                linear = linear_map[most]
                setattr(linear, last, QuantizedTensor.load_from_reader_tensor(tensor))
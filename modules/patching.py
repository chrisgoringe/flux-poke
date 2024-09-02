import torch
from .utils import layer_iteratable_from_string
from .casting import QuantizedTensor, DequantingLinear

from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader


def name_all_linears(layer_stack, layer_list, block_constraint) -> dict[str, dict]:
    name_linear_map:dict[str,torch.nn.Module] = {}

    def all_linears(name:str, module:torch.nn.Module):
        for n,m in module.named_children():
            nm = ".".join((name, n))
            if isinstance(m,torch.nn.Linear):
                if block_constraint is None or block_constraint in nm:
                    name_linear_map[nm] = {"parent":module}
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
            most, last = ".".join(tensor.name.split(".")[:-1]) , tensor.name.split(".")[-1] 
            if most in linear_map:
                linear_map[most][last] = QuantizedTensor.load_from_reader_tensor(tensor)

        for linear in linear_map:
            parent = linear_map[linear]['parent']
            name   = linear.split(".")[-1] 
            dq = DequantingLinear( reader_tensor_weight=linear_map[linear]['weight'], reader_tensor_bias=linear_map[linear].get('bias', None) )
            if verbose: print(f"Patching {linear}")
            setattr(parent, name, dq)


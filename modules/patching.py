import torch
from .utils import layer_iteratable_from_string
from .casting import DequantingLinear

from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader, GGMLQuantizationType

class PatchElement:
    READER_TENSOR = 1

    def __init__(self, fullname:str, parent:torch.nn.Module):
        self.fullname   = fullname
        self.childname  = fullname.split(".")[-1]
        self.parent     = parent
        self.patch_type = None
        self.weight     = None
        self.bias       = None

    def set_tensor(self, tensor_name, tensor):
        if isinstance(tensor, ReaderTensor):
            assert self.patch_type is None or self.patch_type == self.READER_TENSOR
            self.patch_type = self.READER_TENSOR
            if tensor_name=='weight': self.weight = tensor
            elif tensor_name=='bias': self.bias = tensor
            else: raise NotImplementedError()
        else:
            raise NotImplementedError()
    

class PatchingMap:
    def __init__(self):
        self.elements:dict[str, PatchElement] = {}

    def add_linear(self, fullname:str, parent:torch.nn.Module):
        self.elements[fullname] = PatchElement(fullname, parent)

    def add_tensor(self, linear_name:str, tensor_name:str, tensor:ReaderTensor):
        self.elements[linear_name].set_tensor(tensor_name, tensor)

    def get_weight_and_bias(self, linear_name:str) -> tuple[ReaderTensor, ReaderTensor]:
        return ( self.elements[linear_name].weight, self.elements[linear_name].bias )
    
    def get_parent(self, linear_name:str) -> torch.nn.Module:
        return self.elements[linear_name].parent
    
    def get_childname(self, linear_name:str) -> str:
        return self.elements[linear_name].childname
    
    def get_quant_type(self, linear_name:str) -> GGMLQuantizationType:
        return self.elements[linear_name].weight.tensor_type

    def __contains__(self, name):
        return name in self.elements
    
    def __iter__(self): 
        for linear_name in self.elements: yield linear_name



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
            qt:GGMLQuantizationType = linear_map.get_quant_type(linear)
            if verbose: print(f"Patching {linear} to {qt.name}")
            setattr(
                    linear_map.get_parent(linear), 
                    linear_map.get_childname(linear), 
                    DequantingLinear.from_reader_tensors( weight_and_bias=linear_map.get_weight_and_bias(linear) )
                    )



import folder_paths
import comfy
from comfy.model_detection import model_config_from_unet
from comfy.model_management import unet_offload_device, get_torch_device
from safetensors.torch import load_file
import torch
import math, os, logging
from .modules.utils import filepath, load_config, SingletonAddin, layer_iteratable_from_string
from comfy.model_management import DISABLE_SMART_MEMORY
from .modules.gguf_py.gguf import GGMLQuantizationType
from .modules.casting import QuantizedTensor, dequantize_tensor, quantise_tensor
from .modules.utils import FluxFacts
from typing import Union
from functools import partial

relative_to_me = partial(os.path.join, os.path.dirname(__file__))

class LoadTracker(SingletonAddin):
    def __init__(self):
        self.parameters_by_type = {}

    def reset(self):
        self.parameters_by_type = {}

    def track(self, type, shape):
        self.parameters_by_type[type] = self.parameters_by_type.get(type,0) + math.prod(shape)

    def bits_by_type(self, type, default):

        if type in ['bfloat16', 'float16']: return 16
        if type in ['float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz']: return 8
        if type=='Q8_0': return 8
        if type=='Q5_1': return 5
        if type=='Q4_1': return 4
        return default
        
    def total_bits(self, default):
        return sum( self.bits_by_type(t, default)*self.parameters_by_type[t] for t in self.parameters_by_type )
    
    def unreduced_bits(self, default):
        return default * sum( self.parameters_by_type[t] for t in self.parameters_by_type )
    
class Castings:
    casts = []
    default = None
    @classmethod
    def configure(cls, configuration):
        cls.casts = []
        for cast in configuration['casts']:
            layers = [x for x in layer_iteratable_from_string(cast.get('layers', None))]
            blocks = cast.get('blocks', None)
            cast_to = cast.get('castto', 'none')
            cls.casts.append((layers, blocks, cast_to))
        if 'default' in configuration:
            cls.default = configuration['default']
            cls.casts.append((list(range(FluxFacts.last_layer+1)), None, configuration['default']))

    @classmethod
    def get_layer_and_subtype(cls, label) -> int:
        s = label.split(".")
        if s[0]=="double_blocks":
            if s[2].startswith('img'):   return FluxFacts.first_double_layer + int(s[1]), 'img'
            elif s[2].startswith('txt'): return FluxFacts.first_double_layer + int(s[1]), 'txt'
            else:                        return None, None
        elif s[1]=="single_blocks":      return FluxFacts.first_single_layer + int(s[1]), 'x'
        else:                            return None, None

    @classmethod
    def getcast(cls, label) -> str:
        layer, subtype = cls.get_layer_and_subtype(label)
        if layer is None: return cls.default
        for (layers, blocks, cast_to) in cls.casts:
            if (layer in layers) and (blocks is None or blocks==subtype):
                if cast_to == 'none': return None
                if cast_to == 'default': return cls.default
                return cast_to
    
class MixedOps(comfy.ops.disable_weight_init):
    
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight:Union[torch.Tensor,QuantizedTensor] = None
            self.bias:Union[torch.Tensor,QuantizedTensor]   = None
            self.cast:str = None

        def cast_tensor(self, data:torch.Tensor) -> Union[torch.Tensor,QuantizedTensor]:
            if self.cast is None:
                return data
            elif hasattr(GGMLQuantizationType, self.cast):
                gtype = getattr(GGMLQuantizationType, self.cast)
                return quantise_tensor(t=data, gtype=gtype)
            elif hasattr(torch, self.cast):
                return data.to(getattr(torch, self.cast))
            else:
                raise NotImplementedError(self.cast)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if self.cast is None: 
                self.cast = Castings.getcast(prefix)
                if self.cast: 
                    if hasattr(self.cast,'name'):
                        logging.info(f"Casting {prefix} to {self.cast.name}")
                    else:
                        logging.info(f"Casting {prefix} to {self.cast}")

            for k,v in state_dict.items():
                if k[len(prefix):] == "weight":
                    self.weight = self.cast_tensor(v)
                elif k[len(prefix):] == "bias":
                    self.bias = self.cast_tensor(v)
                else:
                    unexpected_keys.append(k) 
                LoadTracker.instance().track(self.cast, v.shape)

            assert self.weight is not None

        def _save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
            if self.weight is not None:
                weight = torch.zeros_like(self.weight, device=torch.device("meta"))
                destination[f"{prefix}weight"] = weight
            if self.bias is not None:
                bias = torch.zeros_like(self.bias, device=torch.device("meta"))
                destination[f"{prefix}bias"] = bias
            return

        def _apply(self, fn):
            if self.weight is not None:
                self.weight = fn(self.weight)
            if self.bias is not None:
                self.bias = fn(self.bias)
            super()._apply(fn)
            return self
        
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            device = None
            if self.weight.device != x.device:
                device = self.weight.device
                self.to(x.device)

            if hasattr(torch, self.cast):
                with torch.autocast(x.device):
                    x = torch.nn.functional.linear(x, self.weight, self.bias)
            else:
                weight = self.dequantize_and_patch(self.weight, match=x)
                bias   = self.dequantize_and_patch(self.bias,   match=x)
                x = torch.nn.functional.linear(x, weight, bias)
                del weight, bias

            if device: self.to(device)
            return x

        def move_patch_to(self, item, match:torch.Tensor):
            if isinstance(item, torch.Tensor):
                return item.to(match.dtype).to(match.device, non_blocking=True)
            elif isinstance(item, tuple):
                return tuple(self.move_patch_to(x, match) for x in item)
            elif isinstance(item, list):
                return [self.move_patch_to(x, match) for x in item]
            else:
                return item

        def dequantize_and_patch(self, tensor:Union[QuantizedTensor, torch.Tensor], match:torch.Tensor):
            if tensor is None: return None

            patch_list = []
            for function, patches, _ in getattr(tensor, "patches", []):
                patch_list += self.move_patch_to(patches, match)

            if isinstance(tensor, QuantizedTensor):
                weight = dequantize_tensor(tensor, match.dtype, match.device)
            else:
                weight = tensor.to(match.dtype).to(match.device)           

            if patch_list: weight = function(patch_list, weight, "dequant.name.unknown")
            return weight                                

class UnetLoaderMixed:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        modelnames = [x for x in folder_paths.get_filename_list("diffusion_models")]
        return {
            "required": {
                "modelname": (modelnames,),
                "casting_file": ("STRING", {"default":""}),
            }
        }

    @classmethod
    def IS_CHANGED(self, modelname, casting_file):
        with open(relative_to_me('settings',casting_file)) as f:
            return modelname+f.read()

    def func(self, modelname, casting_file):
        DISABLE_SMART_MEMORY = True
        Castings.configure(load_config(relative_to_me('settings',casting_file)))

        LoadTracker.instance().reset()
        model = comfy.sd.load_diffusion_model_state_dict(
            load_file(folder_paths.get_full_path("diffusion_models", modelname)), 
            model_options={"custom_operations":MixedOps}
        )
        model.model.cuda()
        mfac =  LoadTracker.instance().total_bits(16) / FluxFacts.bits_at_bf16
        logging.info("Model size reduced to {:>5.2f}%".format(100*mfac))
        model.model.memory_usage_factor *= mfac

        return (model,)

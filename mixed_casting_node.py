import folder_paths
import comfy
from safetensors.torch import load_file
import torch
import json, re, math
from .modules.utils import filepath
from comfy.model_management import DISABLE_SMART_MEMORY

class LoadTracker:
    def __init__(self):
        self.parameters_by_type = {}

    def track(self, type, shape):
        self.parameters_by_type[type] = self.parameters_by_type.get(type,0) + math.prod(shape)

    def bits_by_type(self, type):
        if type in [torch.double, torch.float64]: return 64
        if type in [torch.float, torch.float32]: return 32
        if type in [torch.bfloat16, torch.float16, torch.half]: return 16
        if type in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]: return 8

    def total_bits(self):
        return sum( self.bits_by_type(t)*self.parameters_by_type[t] for t in self.parameters_by_type )
    
class Castings:
    castings = None
    @classmethod
    def load_castings_and_options(cls, filepath):
        with open(filepath) as f: 
            castings = json.load(f)
            model_options = castings.pop('model_options',{})
            cls.castings = [ (re.compile(k), getattr(torch, castings[k])) for k in castings ]
        return model_options
    
class MixedOps(comfy.ops.disable_weight_init):
    FULL_LOAD_BITS = 190422221824
    
    load_tracker:LoadTracker
    @classmethod
    def reset_load_tracking(cls):
        cls.load_tracker = LoadTracker()
    
    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = None
            self.bias   = None

        def type_from_prefix(self, prefix:str):
            for rgx, the_type in Castings.castings:
                if rgx.match(prefix): return the_type
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            my_type = self.type_from_prefix(prefix)
            print(f"{prefix} {my_type}")
            for k,v in state_dict.items():
                if k[len(prefix):] == "weight":
                    self.weight = (v.to(my_type)) # This will also handle size if we're loading something saved pruned...
                elif k[len(prefix):] == "bias":
                    self.bias = (v.to(my_type))
                else:
                    unexpected_keys.append(k) 
                MixedOps.load_tracker.track(my_type, v.shape)

        def _apply(self, fn):
            if self.weight is not None:
                self.weight = fn(self.weight)
            if self.bias is not None:
                self.bias = fn(self.bias)
            super()._apply(fn)
            return self
        
        def forward(self, x):
            with torch.autocast("cuda"):
                return torch.nn.functional.linear(x, self.weight, self.bias)

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
        with open(filepath(casting_file)) as f:
            return f.read()

    def func(self, modelname, casting_file):
        #DISABLE_SMART_MEMORY = True
        model_options = Castings.load_castings_and_options(filepath(casting_file))
        if 'dtype' in model_options: model_options['dtype'] = getattr(torch, model_options['dtype'])
        model_options["custom_operations"] = MixedOps

        MixedOps.reset_load_tracking()
        model = comfy.sd.load_diffusion_model_state_dict(
            load_file(folder_paths.get_full_path("diffusion_models", modelname)), 
            model_options=model_options
        )

        mfac =  MixedOps.load_tracker.total_bits() / MixedOps.FULL_LOAD_BITS
        print(mfac)
        model.model.memory_usage_factor *= mfac

        return (model,)

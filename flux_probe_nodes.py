from safetensors.torch import load_file
from .trackers import InternalsTracker, HiddenStateTracker
from .modifiers import slice_double_block, replace_double_block_mlps
from nodes import UNETLoader
import torch
import os, random
import folder_paths
from functools import partial

filepath = partial(os.path.join,os.path.split(__file__)[0])
    
class AbstractInserter:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": ("MODEL", ), } }    
    def func(self, model, **kwargs):
        self._func(model, **kwargs)
        return (model,)

class AbstractSaver:
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(cls): 
        return { "required": { 
            "image": ("IMAGE", {}), 
            "filename": ("STRING",{"default":cls.DEFAULT}),
            "append": (["yes", "no"], {}),
            } }      
    def func(self, image, filename, append):
        self.CLAZZ.save_all(filepath(filename), append=(append=='yes'))
        return (image,)      

class InsertInternalProbes(AbstractInserter):
    def _func(self, model):
        for i, block in enumerate(model.model.diffusion_model.double_blocks): 
            if isinstance(block, HiddenStateTracker):
                InternalsTracker.inject_internals_tracker(block.wrapped_module, i)
            else:
                InternalsTracker.inject_internals_tracker(block, i)
        print(f"Added {len(InternalsTracker.all_datasets)} callouts")
    
class InternalsSaver(AbstractSaver):
    DEFAULT = "internals.safetensors"
    CLAZZ   = InternalsTracker
    
class InsertHiddenStateProbes(AbstractInserter):
    def _func(self, model):
        model.model.diffusion_model.double_blocks = torch.nn.ModuleList(
            [ HiddenStateTracker(dsb, i) for i, dsb in enumerate(model.model.diffusion_model.double_blocks) ]
        )
        print(f"Tracking {len(HiddenStateTracker.hidden_states)} hidden states")   

class HiddenStatesSaver(AbstractSaver):
    DEFAULT = "hidden_states"
    CLAZZ   = HiddenStateTracker

class ReplaceLayers(AbstractInserter):
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "model": ("MODEL", ), 
        "first_layer":("INT",{"default":0, "man":0, "max":19}), 
        "last_layer":("INT",{"default":0, "man":0, "max":19}),
    }}

    def _func(self, model, first_layer, last_layer):
        for l in range(first_layer, last_layer+1):
            replace_double_block_mlps(block = model.model.diffusion_model.double_blocks[l], 
                                      data = load_file(filepath("retrained_layers",f"{l}.safetensors")))
            

class LoadPrunedFluxModel(UNETLoader):
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("model","masked_count")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name": (folder_paths.get_filename_list("unet"), ),
        "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
        "file": ("STRING", {"default":"internals.safetensors"}), 
        "threshold": ("INT", {"default":-1}),
        "first_layer": ("INT", {"default":0, "min":0, "max":19}),
        "last_layer": ("INT", {"default":0, "min":0, "max":19}),
        } }    
    
    def __init__(self):
        super().__init__()
        self.masked_out = 0
    
    def mask_from(self, data, threshold):
        l = list( d>threshold for d in data )
        self.masked_out += len(data) - sum(l)
        return l

    def func(self, unet_name, weight_dtype, file, threshold, first_layer, last_layer):
        model = self.load_unet(unet_name, weight_dtype)[0]
        all_data = load_file(filepath(file))
        self.masked_out = 0
        for i in range(first_layer, last_layer+1):
            slice_double_block(block    = model.model.diffusion_model.double_blocks[i], 
                               img_mask = self.mask_from(all_data[f"double-img-{i}"], threshold), 
                               txt_mask = self.mask_from(all_data[f"double-txt-{i}"], threshold))
        return (model,str(int(self.masked_out)))    
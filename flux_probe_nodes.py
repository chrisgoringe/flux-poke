from safetensors.torch import load_file
from .trackers import InternalsTracker, slice_double_block, HiddenStateTracker
from nodes import UNETLoader
import torch
import os
ROOT = os.path.split(__file__)[0]
    
class AbstractInserter:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": ("MODEL", ), } }    

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
        self.CLAZZ.save_all(os.path.join(ROOT,filename), append=(append=='yes'))
        return (image,)      

class InsertInternalProbes(AbstractInserter):
    def func(self, model):
        for i, block in enumerate(model.model.diffusion_model.double_blocks): 
            if isinstance(block, HiddenStateTracker):
                InternalsTracker.inject_internals_tracker(block.wrapped_module, i)
            else:
                InternalsTracker.inject_internals_tracker(block, i)
        print(f"Added {len(InternalsTracker.all_datasets)} callouts")
        return (model,)
    
class InternalsSaver(AbstractSaver):
    DEFAULT = "internals.safetensors"
    CLAZZ   = InternalsTracker
    
class InsertHiddenStateProbes(AbstractInserter):
    def func(self, model):
        model.model.diffusion_model.double_blocks = torch.nn.ModuleList(
            [ HiddenStateTracker(dsb, i) for i, dsb in enumerate(model.model.diffusion_model.double_blocks) ]
        )
        print(f"Tracking {len(HiddenStateTracker.hidden_states)} hidden states")
        return (model,)    

class HiddenStatesSaver(AbstractSaver):
    DEFAULT = "hidden_states"
    CLAZZ   = HiddenStateTracker

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
        } }    
    
    def __init__(self):
        super().__init__()
        self.masked_out = 0
    
    def mask_from(self, data, threshold):
        l = list( d>threshold for d in data )
        self.masked_out += len(data) - sum(l)
        return l

    def func(self, unet_name, weight_dtype, file, threshold):
        model = self.load_unet(unet_name, weight_dtype)[0]
        all_data = load_file(os.path.join(ROOT,file))
        self.masked_out = 0
        for i, block in enumerate(model.model.diffusion_model.double_blocks): 
            img_data = all_data[f"double-img-{i}"]
            txt_data = all_data[f"double-txt-{i}"]
            slice_double_block(block, self.mask_from(img_data, threshold), self.mask_from(txt_data, threshold))
        return (model,str(int(self.masked_out)))    
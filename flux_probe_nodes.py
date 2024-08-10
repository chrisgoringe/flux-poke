from safetensors.torch import load_file
from .modules.trackers import InternalsTracker, HiddenStateTracker
from .modules.modifiers import slice_double_block, replace_double_block_mlps, get_mask
from nodes import UNETLoader
import torch
import os
import folder_paths
from functools import partial
from comfy.model_management import cleanup_models

filepath = partial(os.path.join,os.path.split(__file__)[0])
    
class AbstractModelModifier:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": ("MODEL", ), } }    
    def func(self, model, **kwargs):
        m = model.clone()
        print(self._func(m, **kwargs))
        model = None
        cleanup_models()
        return (m,)
 
class InsertInternalProbes(AbstractModelModifier):
    def _func(self, model):
        for i, block in enumerate(model.model.diffusion_model.double_blocks): 
            if isinstance(block, HiddenStateTracker):
                InternalsTracker.inject_internals_tracker(block.wrapped_module, i)
            else:
                InternalsTracker.inject_internals_tracker(block, i)
        #for i, block in enumerate(model.model.diffusion_model.single_blocks): 
        #    if isinstance(block.linear2, torch.nn.Sequential): block.linear2 = block.linear2[1]
        #    InternalsTracker.inject_internals_tracker(block, i+len(model.model.diffusion_model.double_blocks))

        return f"Added {len(InternalsTracker.all_datasets)} callouts"

class InsertHiddenStateProbes(AbstractModelModifier):
    def _func(self, model):
        model.model.diffusion_model.double_blocks = torch.nn.ModuleList(
            [ HiddenStateTracker(dsb, i) for i, dsb in enumerate(model.model.diffusion_model.double_blocks) ]
        )
        #n_double = len(model.model.diffusion_model.double_blocks)
        #model.model.diffusion_model.single_blocks = torch.nn.ModuleList(
        #    [ HiddenStateTracker(dsb, i+n_double) for i, dsb in enumerate(model.model.diffusion_model.single_blocks) ]
        #)        
        return f"Tracking {len(HiddenStateTracker.hidden_states)} hidden states"  


class AbstractSaver:
    RETURN_TYPES = ("LATENT",)
    OUTPUT_NODE = True
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(cls): 
        return { "required": { 
            "latent": ("LATENT", {}), 
            "filename": ("STRING",{"default":cls.DEFAULT}),
            } }      
    def func(self, latent, filename):
        self.SAVER(filepath(filename))
        return (latent,)     
    
class InternalsSaver(AbstractSaver):
    DEFAULT = "internals.safetensors"
    SAVER   = InternalsTracker.save_all
    
class HiddenStatesSaver(AbstractSaver):
    DEFAULT = "hidden_states"
    SAVER   = HiddenStateTracker.save_all



class ReplaceLayers(UNETLoader):
    RETURN_TYPES = ("MODEL","STRING",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name":            (folder_paths.get_filename_list("unet"), ),
        "weight_dtype":         (["default", "fp8_e4m3fn", "fp8_e5m2"],),
        "replacement_directory":("STRING",{"default":"retrained_layers"}),
        "first_layer":          ("INT",{"default":0,  "min":0, "max":18}), 
        "last_layer":           ("INT",{"default":18, "min":0, "max":18}),
    }}

    def func(self, unet_name, weight_dtype, replacement_directory, first_layer, last_layer):
        model = self.load_unet(unet_name, weight_dtype)[0]
        dbs   = model.model.diffusion_model.double_blocks
        for l in range(first_layer, last_layer+1):
            if os.path.exists(file:=filepath(replacement_directory,f"{l}.safetensors")):
                replace_double_block_mlps(block = dbs[l], data = load_file(file))
            else:
                print(f"{file} not found")
        img_masked = sum( (12288-l.img_mlp[0].out_features) for l in dbs )
        txt_masked = sum( (12288-l.txt_mlp[0].out_features) for l in dbs )
        return (model,f"img{img_masked}_txt{txt_masked}",)


class LoadPrunedFluxModel(UNETLoader):
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("model","masked_count")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name":    (folder_paths.get_filename_list("unet"), ),
        "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
        "file":         ("STRING", {"default":"internals.safetensors"}), 
        "img_cut":      ("INT", {"default":2000, "min":0, "max":12288}),
        "txt_cut":      ("INT", {"default":2000, "min":0, "max":12288}),
        "first_layer":  ("INT", {"default":0,  "min":0, "max":18}),
        "last_layer":   ("INT", {"default":18, "min":0, "max":18}),
        } }    

    def func(self, unet_name, weight_dtype, file, first_layer, last_layer, img_cut, txt_cut):
        return self._func(unet_name, weight_dtype, file, first_layer, last_layer, img_cut, txt_cut)

    def _func(self, unet_name, weight_dtype, file, first_layer, last_layer, img_threshold=None, txt_threshold=None, img_cut=None, txt_cut=None):
        model = self.load_unet(unet_name, weight_dtype)[0]
        all_data = load_file(filepath(file))
        img_masked, txt_masked = 0, 0
        for i in range(first_layer, last_layer+1):
            block = model.model.diffusion_model.double_blocks[i]
            img_mask = get_mask(all_data[f"double-img-{i}"], img_threshold, img_cut)
            txt_mask = get_mask(all_data[f"double-txt-{i}"], txt_threshold, txt_cut)
            img_masked += sum(not x for x in img_mask) 
            txt_masked += sum(not x for x in txt_mask)
            slice_double_block(block = block, img_mask = img_mask, txt_mask = txt_mask )
        print(f"Total of {img_masked} img lines and {txt_masked} txt lines cut")
        return (model,f"img{img_masked}_txt{txt_masked}",)
    
class LoadPrunedFluxModelThreshold(LoadPrunedFluxModel):
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name":     (folder_paths.get_filename_list("unet"), ),
        "weight_dtype":  (["default", "fp8_e4m3fn", "fp8_e5m2"],),
        "file":          ("STRING", {"default":"internals.safetensors"}), 
        "img_threshold": ("INT", {"default":2000, "min":0}),
        "txt_threshold": ("INT", {"default":2000, "min":0}),
        "first_layer":   ("INT", {"default":0,  "min":0, "max":18}),
        "last_layer":    ("INT", {"default":18, "min":0, "max":18}),
        } }    
    
    def func(self, unet_name, weight_dtype, file, first_layer, last_layer, img_threshold, txt_threshold):
        return self._func(unet_name, weight_dtype, file, first_layer, last_layer, img_threshold, txt_threshold)
    
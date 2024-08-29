from safetensors.torch import load_file
from .modules.trackers import InternalsTracker, HiddenStateTracker
from .modules.modifiers import slice_double_block, replace_double_block_mlps, get_mask
from nodes import UNETLoader
import torch
import os
import folder_paths
from functools import partial
from comfy.model_management import cleanup_models
from comfy.ldm.flux.model import Flux

filepath = partial(os.path.join,os.path.split(__file__)[0])
 
class InsertProbes:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "model": ("MODEL", ), 
        "track_hidden_states" : (["yes","no"],{}),
        "just_in_out" : (["no","yes"],{}),
        "track_internals" : (["yes","no"],{}),
    } }  

    def func(self, model, track_hidden_states, just_in_out, track_internals):
        m = model.clone()
        flux:Flux = m.model.diffusion_model
        n_double = len(flux.double_blocks)

        if track_hidden_states=='yes':
            if just_in_out=="yes":
                HiddenStateTracker.set_mode(all_in_one=True)
                #db_list = [ HiddenStateTracker(diffusion_model.double_blocks[0], 0,store_input=True, store_output=False), ] + [db for db in diffusion_model.double_blocks[1:]]
                flux.double_blocks[0] = HiddenStateTracker(flux.double_blocks[0], 0,store_input=True, store_output=False)
                #torch.nn.ModuleList(db_list)
                flux.single_blocks[-1] = HiddenStateTracker(flux.single_blocks[-1], 56, store_input=False, store_output=True)

            else:
                HiddenStateTracker.set_mode(all_in_one=False)
                def replace_list(list_name, first_index):
                    new_list = [ HiddenStateTracker(dsb, i+first_index) for i, dsb in enumerate(getattr(flux,list_name)) ]
                    setattr(flux, list_name, torch.nn.ModuleList(new_list))

                replace_list('double_blocks',0)
                replace_list('single_blocks', n_double)
            print (f"Tracking {len(HiddenStateTracker.hidden_states)} hidden states")  

        if track_internals=='yes':
            def inject_internals_tracker(list_name, first_index):
                for i, block in enumerate(getattr(flux, list_name)):
                    InternalsTracker.inject_internals_tracker(block.wrapped_module if isinstance(block, HiddenStateTracker) else block, i+first_index)

            inject_internals_tracker('double_blocks', 0)
            inject_internals_tracker('single_blocks', n_double)
            print( f"Added {len(InternalsTracker.all_datasets)} callouts" )

        return (m,)

class SaveProbeData:
    RETURN_TYPES = ("LATENT",)
    OUTPUT_NODE = True
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(cls): 
        return { "required": { 
            "latent": ("LATENT", {}), 
            "filename_for_internals": ("STRING",{"default":"internals.safetensors"}),
            "repo_id_for_hidden": ("STRING",{"default":"ChrisGoringe/fi"}),
            } }      
    def func(self, latent, filename_for_internals, repo_id_for_hidden):
        InternalsTracker.save_all(filepath(filename_for_internals))
        HiddenStateTracker.save_all(repo_id_for_hidden)
        return (latent,)     

class ReplaceLayers(UNETLoader):
    RETURN_TYPES = ("MODEL","STRING",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name":            (folder_paths.get_filename_list("diffusion_models"), ),
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
        "unet_name":    (folder_paths.get_filename_list("diffusion_models"), ),
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
        all_data = load_file(filepath("data",file))
        img_masked, txt_masked = 0, 0
        for i in range(first_layer, last_layer+1):
            block = model.model.diffusion_model.double_blocks[i]
            img_mask, _ = get_mask(all_data[f"double-img-{i}"], remove_below=img_threshold, remove_count=img_cut)
            txt_mask, _ = get_mask(all_data[f"double-txt-{i}"], remove_below=txt_threshold, remove_count=txt_cut)
            img_masked += sum(not x for x in img_mask) 
            txt_masked += sum(not x for x in txt_mask)
            slice_double_block(block = block, img_mask = img_mask, txt_mask = txt_mask )
        print(f"Total of {img_masked} img lines and {txt_masked} txt lines cut")
        return (model,f"img{img_masked}_txt{txt_masked}",)
    
class LoadPrunedFluxModelThreshold(LoadPrunedFluxModel):
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "unet_name":     (folder_paths.get_filename_list("diffusion_models"), ),
        "weight_dtype":  (["default", "fp8_e4m3fn", "fp8_e5m2"],),
        "file":          ("STRING", {"default":"internals.safetensors"}), 
        "img_threshold": ("INT", {"default":2000, "min":0}),
        "txt_threshold": ("INT", {"default":2000, "min":0}),
        "first_layer":   ("INT", {"default":0,  "min":0, "max":18}),
        "last_layer":    ("INT", {"default":18, "min":0, "max":18}),
        } }    
    
    def func(self, unet_name, weight_dtype, file, first_layer, last_layer, img_threshold, txt_threshold):
        return self._func(unet_name, weight_dtype, file, first_layer, last_layer, img_threshold, txt_threshold)
    
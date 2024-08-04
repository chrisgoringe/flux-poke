from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from typing import Union
from safetensors.torch import save_file, load_file

class InternalsTracker(torch.nn.Module):
    all_datasets = {}
    def __init__(self,label:str):
        super().__init__()
        self.label = label
        self.all_datasets[self.label] = 0
        
    def forward(self, x:torch.Tensor):
        self.all_datasets[self.label] += torch.sum((x>0),dim=(0,1)).cpu()
        return x
    
    @classmethod
    def reset_all(cls):
        for k in cls.all_datasets: cls.all_datasets[k]=0

    @classmethod
    def save_all(cls, filepath, then_reset=False):
        save_file(cls.all_datasets, filename=filepath)
        if then_reset: cls.reset_all()

class InternalsMasker(torch.nn.Module):
    ONES = None
    ZERO = None
    def __init__(self,mask:list[bool]):
        super().__init__()
        self.set_mask(mask)

    def set_mask(self, mask:list[bool]):
        if InternalsMasker.ONES==None:
            l = len(mask)
            InternalsMasker.ONES = torch.ones((l))
            InternalsMasker.ZERO = torch.zeros((l))
        self.mult = torch.where(torch.tensor(mask), InternalsMasker.ONES, InternalsMasker.ZERO).cuda()

    def forward(self, x:torch.Tensor):
        return x * self.mult    

class WrappedLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Module, label:str, last_n:int):
        super().__init__()
        self.linear = linear
        self.callout = InternalsTracker(label)
        self.last_n = last_n

    def forward(self, x:torch.Tensor):
        self.callout(x[...,-self.last_n:])
        return self.linear(x)

class MaskedLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Module, mask:list[bool]):
        super().__init__()
        self.linear = linear
        self.callout = InternalsMasker(mask)
        self.last_n = len(mask)

    def set_mask(self, mask:list[bool]):
        self.callout.set_mask(mask)

    def forward(self, x:torch.Tensor):
        x = torch.cat((x[...,:-self.last_n], self.callout(x[...,-self.last_n:])), dim=-1)
        return self.linear(x)

def inject_internals_tracker(block:Union[DoubleStreamBlock, SingleStreamBlock], index:int):
    if isinstance(block, DoubleStreamBlock):
        if isinstance( block.img_mlp[2], InternalsTracker ): return
        block.img_mlp.insert(2, InternalsTracker(f"double-img-{index}"))
        block.txt_mlp.insert(2, InternalsTracker(f"double-txt-{index}"))
        pass
    elif isinstance(block, SingleStreamBlock):
        block.linear2 = WrappedLinear(block.linear2, f"{index}-single", block.mlp_hidden_dim)
        pass
    else:
        raise NotImplementedError()
    
def inject_masker(block:Union[DoubleStreamBlock, SingleStreamBlock], mask:list[bool], mask2:list[bool]=None):
    if isinstance(block, DoubleStreamBlock):
        if isinstance( block.img_mlp[2], InternalsMasker ): 
            block.img_mlp[2].set_mask(mask)
            block.txt_mlp[2].set_mask(mask)
        else:
            block.img_mlp.insert(2, InternalsMasker(mask))
            block.txt_mlp.insert(2, InternalsMasker(mask2))
        pass
    elif isinstance(block, SingleStreamBlock):
        if isinstance(block.linear2, MaskedLinear):
            block.linear2.set_mask(mask)
        else:
            block.linear2 = MaskedLinear(block.linear2, mask)
        pass
    else:
        raise NotImplementedError()
    
class FluxWatcher:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model": ("MODEL", ), } }    

    def func(self, model):
        for i, block in enumerate(model.model.diffusion_model.double_blocks): inject_internals_tracker(block, i)
        for i, block in enumerate(model.model.diffusion_model.single_blocks): inject_internals_tracker(block, i)
        print(f"Added {len(InternalsTracker.all_datasets)} callouts")
        return (model,)
    
class FluxMasker:
    RETURN_TYPES = ("MODEL","STRING")
    RETURN_NAMES = ("model","masked_count")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "model": ("MODEL", ), 
        "file": ("STRING", {"default":"internals.safetensors"}), 
        "threshold": ("INT", {"default":-1}),
        } }    
    
    def __init__(self):
        self.masked_out = 0
    
    def mask_from(self, data, threshold):
        l = list( d>threshold for d in data )
        self.masked_out += len(data) - sum(l)
        return l

    def func(self, model, file, threshold):
        all_data = load_file(file)
        self.masked_out = 0
        for i, block in enumerate(model.model.diffusion_model.double_blocks): 
            img_data = all_data[f"double-img-{i}"]
            txt_data = all_data[f"double-txt-{i}"]
            inject_masker(block, self.mask_from(img_data, threshold), self.mask_from(txt_data, threshold))
        for i, block in enumerate(model.model.diffusion_model.single_blocks): 
            data = all_data[f"{i}-single"]
            inject_masker(block, self.mask_from(data, threshold))
        return (model,str(int(self.masked_out)))    
    
class FluxInternalsSaver:
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): 
        return { "required": { 
            "image": ("IMAGE", {}), 
            "filename": ("STRING",{"default":"internals.safetensors"}),
            "then_reset": (["no","yes"], {}),
            } }    

    def func(self, image, filename, then_reset):
        InternalsTracker.save_all(filename, then_reset=(then_reset=='yes'))
        return (image,)
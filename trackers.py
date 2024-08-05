from comfy.ldm.flux.layers import DoubleStreamBlock
import torch
from datasets import Dataset, concatenate_datasets
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import os

class HiddenStateTracker(torch.nn.Module):
    hidden_states: dict[int, list[tuple[torch.Tensor]]] = {}  # index i is the hidden states *before* layer i
    def __init__(self, dsb_to_wrap:DoubleStreamBlock, layer:int, store_before:bool=None, store_after:bool=True):
        super().__init__()
        self.store_before   = store_before if store_before is not None else (layer==0)
        self.store_after    = store_after
        self.wrapped_module = dsb_to_wrap
        self.layer          = layer
        if layer   not in self.hidden_states and self.store_before: self.hidden_states[layer]   = []
        if layer+1 not in self.hidden_states and self.store_after:  self.hidden_states[layer+1] = [] 

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        if self.store_before: self.hidden_states[self.layer].append( (img.cpu(), txt.cpu()) )
        img, txt = self.wrapped_module(img, txt, vec, pe)
        self.hidden_states[self.layer + 1].append( (img.cpu(), txt.cpu()) )
        return img, txt
    
    @classmethod
    def reset_all(cls):
        for k in cls.hidden_states: cls.hidden_states[k]=[]

    @classmethod
    def save_all(cls, filepath, append=True):
        length = min( len(cls.hidden_states[k] for k in cls.hidden_states) )
        def gen():
            for index in range(length):
                yield { k:cls.hidden_states[k][index] for k in cls.hidden_states } 
        dataset:Dataset = Dataset.from_generator(gen)
        if append and os.path.exists(filepath):
            dataset:Dataset = concatenate_datasets([Dataset.load_from_disk(filepath), dataset])
            with TemporaryDirectory() as tmpdirname:
                dataset.save_to_disk(filename=tmpdirname)
                dataset = Dataset.load_from_disk(tmpdirname)
        dataset.save_to_disk(filename=filepath)
        cls.reset_all()

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
    def save_all(cls, filepath, append=True):
        if append and os.path.exists(filepath):
            old = load_file(filename=filepath)
            for k in cls.all_datasets: cls.all_datasets[k] += old.pop(k,0)
            for k in old: cls.all_datasets[k] = old[k]
        save_file(cls.all_datasets, filename=filepath)
        cls.reset_all()

    @classmethod
    def inject_internals_tracker(cls,block:DoubleStreamBlock, index:int):
        if isinstance( block.img_mlp[2], InternalsTracker ): return
        block.img_mlp.insert(2, InternalsTracker(f"double-img-{index}"))
        block.txt_mlp.insert(2, InternalsTracker(f"double-txt-{index}"))

def prune(t:torch.Tensor, mask):
    t, transpose = (t.T, True) if len(t.shape)==2 and t.shape[1] == len(mask) else (t, False)
    if t.shape[0] != len(mask): return t
    t = torch.stack( list(t[i,...] for i,m in enumerate(mask) if m ) )
    return t.T if transpose else t

def new_mlp(old_mlp, mask):
    assert len(mask)==old_mlp[0].weight.shape[0]
    if all(mask): return old_mlp

    clazz       = old_mlp[0].__class__
    hidden_size = old_mlp[0].weight.shape[1]
    dtype       = old_mlp[0].weight.dtype
    device      = old_mlp[0].weight.device
    hidden_dim  = sum(mask)

    mlp = torch.nn.Sequential(
        clazz(hidden_size, hidden_dim, bias=True, dtype=dtype, device=device),
        torch.nn.GELU(approximate="tanh"),
        clazz(hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),        
    )
    old_sd = old_mlp.state_dict()
    sd = { k:prune(old_sd[k],mask) for k in old_sd }
    mlp.load_state_dict(sd)
    return mlp

def slice_double_block(block:DoubleStreamBlock, img_mask:list[bool], txt_mask:list[bool]):
    block.img_mlp = new_mlp(block.img_mlp, img_mask)
    block.txt_mlp = new_mlp(block.txt_mlp, txt_mask)

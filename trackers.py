from comfy.ldm.flux.layers import DoubleStreamBlock
import torch
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import os, random

class DiskCache:
    def __init__(self):
        self.directory = TemporaryDirectory()
        self.number = 0

    def append(self, data:dict[str,torch.Tensor]):
        save_file(data, os.path.join(self.directory.name, str(self.number)))
        self.number += 1

    def __len__(self): 
        return self.number

    def __getitem__(self, i): 
        return load_file(os.path.join(self.directory.name, str(i)))

class HiddenStateTracker(torch.nn.Module):
    hidden_states: dict[int, DiskCache] = {}  # index i is the hidden states *before* layer i
    active = False
    def __init__(self, dsb_to_wrap:DoubleStreamBlock, layer:int, store_before:bool=None, store_after:bool=True):
        super().__init__()
        self.store_before   = store_before if store_before is not None else (layer==0)
        self.store_after    = store_after
        self.wrapped_module = dsb_to_wrap
        self.layer          = layer
        self.is_master      = (layer==0)
        if layer   not in self.hidden_states and self.store_before: self.hidden_states[layer]   = DiskCache()
        if layer+1 not in self.hidden_states and self.store_after:  self.hidden_states[layer+1] = DiskCache()

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        if self.is_master: HiddenStateTracker.active = (random.random() < 0.05)  # 5% of steps captured
        if HiddenStateTracker.active:
            if self.store_before: self.hidden_states[self.layer].append( {"img":img.cpu(), "txt":txt.cpu(), "vec":vec.cpu(), "pe":pe.cpu()} )
            img, txt = self.wrapped_module(img, txt, vec, pe)
            self.hidden_states[self.layer + 1].append( {"img":img.cpu(), "txt":txt.cpu(), "vec":vec.cpu(), "pe":pe.cpu()} )
            return img, txt
        else: return self.wrapped_module(img, txt, vec, pe)
    
    @classmethod
    def reset_all(cls):
        for k in cls.hidden_states: cls.hidden_states[k] = DiskCache()

    @classmethod
    def save_all(cls, filepath, append=True):
        if not os.path.exists(filepath): os.makedirs(filepath, exist_ok=True)
        if not append: raise NotImplementedError() # need to empty the directory
        length = min( [len(cls.hidden_states[k]) for k in cls.hidden_states] )
        def gen():
            for index in range(length):
                data = {}
                for k in cls.hidden_states:
                    for kk in ['img','txt','vec','pe']:
                        data[f"{k}-{kk}"] = cls.hidden_states[k][index][kk]
                yield data
        for datum in gen():
            label = f"{random.randint(1000000,9999999)}.safetensors"
            save_file(datum, os.path.join(filepath, label))
            
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

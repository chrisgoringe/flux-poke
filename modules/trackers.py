from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import os, random
from typing import Union

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
    '''
    Wraps a DoubleStreamBlock. If the index is zero, this is the master block. The master block turns the trackers 
    on for step_fraction of steps (default 0.05) so we capture on average one rando step per image
    '''
    hidden_states: dict[int, DiskCache] = {}  # index i is the hidden states *before* layer i
    active = False
    def __init__(self, block_to_wrap:Union[DoubleStreamBlock, SingleStreamBlock], layer:int, store_before:bool=None, store_after:bool=True, step_fraction:float=0.05):
        super().__init__()
        self.store_before   = store_before if store_before is not None else (layer==0)
        self.store_after    = store_after
        self.wrapped_module = block_to_wrap
        self.layer          = layer
        self.is_master      = (layer==0)
        self.step_fraction  = step_fraction
        if layer   not in self.hidden_states and self.store_before: self.hidden_states[layer]   = DiskCache()
        if layer+1 not in self.hidden_states and self.store_after:  self.hidden_states[layer+1] = DiskCache()

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        if self.is_master: HiddenStateTracker.active = (random.random() < self.step_fraction)  
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
    def __init__(self,label:str, keep_last=12288):
        super().__init__()
        self.label = label
        self.all_datasets[self.label] = 0
        self.keep_last = keep_last
        
    def forward(self, x:torch.Tensor):
        self.all_datasets[self.label] += torch.sum((x>0),dim=(0,1)).cpu()[-self.keep_last:]
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
    def inject_internals_tracker(cls,block:Union[DoubleStreamBlock, SingleStreamBlock], index:int):
        if isinstance( block.img_mlp[2], InternalsTracker ): return

        if isinstance(block, DoubleStreamBlock):
            block.img_mlp.insert(2, InternalsTracker(f"double-img-{index}"))
            block.txt_mlp.insert(2, InternalsTracker(f"double-txt-{index}"))
        elif isinstance(block, SingleStreamBlock):
            block.linear2 = torch.nn.Sequential([
                InternalsTracker((f"single-{index}")),
                block.linear2
            ])

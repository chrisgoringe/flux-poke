from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import os, random, threading, queue, sys, time
from typing import Union
from .hffs import HFFS
from .utils import SingletonAddin

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

class UploadThread(SingletonAddin):
    hffs = HFFS("ChrisGoringe/fi")
    def __init__(self):
        self.queue  = queue.SimpleQueue()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                (label, datum) = self.queue.get()
                print(f"Save {label}...")
                while not self.hffs.save_file(label, datum): 
                    print("Hit hffs rate limits...")
                    time.sleep(15) 
                print(f"qsize returns {self.queue.qsize()}")
            except:
                print(sys.exc_info())

class HiddenStateTracker(torch.nn.Module):
    '''
    Wraps a SingleStreamBlock or DoubleStreamBlock. 
    '''
    SAVE_EVERY = 19
    NEXT_SAVE_COUNTER = 0

    hidden_states: dict[int, DiskCache] = {}  # index i is the hidden states *before* layer i
    active = False
    queue = UploadThread.instance().queue
    def __init__(self, block_to_wrap:Union[DoubleStreamBlock, SingleStreamBlock], layer:int, is_master=None, store_input=None):
        '''
        Wraps `block_to_wrap` and, when active, stores the hidden states.
        `layer` is used as an index
        `is_master` (default None, in which case set to True iff `layer`==0): there should be exactly one master, which turns all trackers on or off
        `store_input` (default None, in which case set to True iff `layer`==0). If True, this tracker needs to store input values as well as output.
        Normally false because the inputs have been stored already as the previous layer's outputs.
        '''
        super().__init__()
        self.store_input    = store_input if store_input is not None else (layer==0)
        self.wrapped_module = block_to_wrap
        self.layer          = layer
        self.is_master      = is_master if is_master is not None else (layer==0)
        self.is_double      = isinstance(block_to_wrap, DoubleStreamBlock)
        if layer   not in self.hidden_states and self.store_input: self.hidden_states[layer]   = DiskCache()
        if layer+1 not in self.hidden_states:                      self.hidden_states[layer+1] = DiskCache()

    def forward(self, *args, **kwargs): # img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        out = self.wrapped_module(*args, **kwargs)

        if self.is_master: 
            HiddenStateTracker.active = (HiddenStateTracker.NEXT_SAVE_COUNTER == 0)
            HiddenStateTracker.NEXT_SAVE_COUNTER = (HiddenStateTracker.NEXT_SAVE_COUNTER + 1) % HiddenStateTracker.SAVE_EVERY

        if HiddenStateTracker.active:
            if self.store_input:
                if self.is_double:
                    self.hidden_states[self.layer].append( { "img":kwargs['img'].cpu(), "txt":kwargs['txt'].cpu(), "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )
                else:
                    self.hidden_states[self.layer].append( { "x":args[0].cpu(),                                    "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )
            if self.is_double: 
                self.hidden_states[self.layer + 1].append( { "img":out[0].cpu(),        "txt":out[1].cpu(),        "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )
            else:                
                self.hidden_states[self.layer + 1].append( { "x":out.cpu(),                                        "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )

        return out

    @classmethod
    def reset_all(cls):
        for k in cls.hidden_states: cls.hidden_states[k] = DiskCache()

    @classmethod
    def save_all(cls, repo_id):
        UploadThread.hffs.set_repo_id(repo_id)
        length = min( [len(cls.hidden_states[k]) for k in cls.hidden_states] )

        def label(r, layer_from, layer_to): return "{:0>7}/{:0>2}-{:0>2}.safetensors".format(r, layer_from, layer_to)
        division = (( 0, 6), ( 7,13), (14,18), (19,25), (26,31), (32,37), (38,43), (44,50), (51,57))     
        for index in range(length):
            r = random.randint(1000000,9999999)
            for (layer_from, layer_to) in division:
                datum = {}
                for layer_index in range(layer_from, layer_to+1):
                    for k in cls.hidden_states[layer_index][index]: datum["{:0>2}".format(layer_index)+"-{k}"] = cls.hidden_states[layer_index][index][k]
                cls.queue.put((label(r,layer_from, layer_to), datum))
            
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

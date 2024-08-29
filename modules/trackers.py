from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from safetensors.torch import save_file, load_file
from tempfile import TemporaryDirectory
import os, random, threading, queue, sys, time
from typing import Union
from .hffs import HFFS
from .utils import SingletonAddin, Batcher
from .generated_dataset import MERGE_SIZE

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
    
from server import PromptServer
from aiohttp import web
routes = PromptServer.instance.routes
@routes.get('/upload_queue')
async def upload_queue(r):
    return web.json_response({"upload_queue":HiddenStateTracker.queue.qsize()})

@routes.get('/download_internals')
async def download_internals(r):
    try:
        return web.json_response(InternalsTracker.download_internals())
    except:
        return web.json_response({"result":f"{sys.exc_info()[1]}"})

@routes.get('/upload_internals')
async def upload_internals(r):
    try:
        return web.json_response(InternalsTracker.upload_internals())
    except:
        return web.json_response({"result":f"{sys.exc_info()[1]}"})



class UploadThread(SingletonAddin):
    hffs = HFFS("ChrisGoringe/fi2")
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


class MergingUploadThread(SingletonAddin):

    def __init__(self):
        self.queue  = queue.SimpleQueue()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        merge_list = []
        while True:
            try:
                merge_list.append( self.queue.get() )
                if len(merge_list) == MERGE_SIZE:
                    merged_label = merge_list[0][0]
                    merged_datum = {}
                    for i, (label, datum) in enumerate(merge_list):
                        for k in datum:
                            merged_datum[f"{i:0>2}_{k}"] = datum[k]
                    print(f"Save {merged_label}...")
                    while not UploadThread.hffs.save_file(merged_label, merged_datum): 
                        print("Hit hffs rate limits...")
                        time.sleep(15) 
                    print(f"qsize returns {self.queue.qsize()}")
                    merge_list = []
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

    @classmethod
    def set_mode(cls, all_in_one):
        Batcher.set_mode(all_in_one)
        if all_in_one: cls.queue = MergingUploadThread.instance().queue
        else:          cls.queue = UploadThread.instance().queue

    def __init__(self, block_to_wrap:Union[DoubleStreamBlock, SingleStreamBlock], layer:int, is_master=None, store_input=None, store_output=True):
        '''
        Wraps `block_to_wrap` and, when active, stores the hidden states.
        `layer` is used as an index
        `is_master` (default None, in which case set to True iff `layer`==0): there should be exactly one master, which turns all trackers on or off
        `store_input` (default None, in which case set to True iff `layer`==0). If True, this tracker needs to store input values as well as output.
        Normally false because the inputs have been stored already as the previous layer's outputs.
        '''
        super().__init__()
        self.store_input    = store_input if store_input is not None else (layer==0)
        self.store_output   = store_output
        self.wrapped_module = block_to_wrap
        self.layer          = layer
        self.is_master      = is_master if is_master is not None else (layer==0)
        self.is_double      = isinstance(block_to_wrap, DoubleStreamBlock)
        if layer   not in self.hidden_states and self.store_input:  self.hidden_states[layer]   = DiskCache()
        if layer+1 not in self.hidden_states and self.store_output: self.hidden_states[layer+1] = DiskCache()

    def forward(self, *args, **kwargs): # img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        
        if self.is_master: 
            HiddenStateTracker.active = (HiddenStateTracker.NEXT_SAVE_COUNTER == 0)
            HiddenStateTracker.NEXT_SAVE_COUNTER = (HiddenStateTracker.NEXT_SAVE_COUNTER + 1) % HiddenStateTracker.SAVE_EVERY

        if HiddenStateTracker.active and self.store_input:
            if self.is_double:
                self.hidden_states[self.layer].append( { "img":kwargs['img'].cpu(), "txt":kwargs['txt'].cpu(), "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )
            else:
                self.hidden_states[self.layer].append( { "x":args[0].cpu(),                                    "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )

        out = self.wrapped_module(*args, **kwargs)

        if HiddenStateTracker.active and self.store_output:
            if self.is_double: 
                self.hidden_states[self.layer+1].append( { "img":out[0].cpu(),      "txt":out[1].cpu(),        "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )
            else:                
                self.hidden_states[self.layer+1].append( { "x":out.cpu(),                                      "vec":kwargs['vec'].cpu(), "pe":kwargs['pe'].cpu() } )

        return out

    @classmethod
    def reset_all(cls):
        for k in cls.hidden_states: cls.hidden_states[k] = DiskCache()

    @classmethod
    def save_all(cls, repo_id):
        length = min( [len(cls.hidden_states[k]) for k in cls.hidden_states] )
        if not length: return
        UploadThread.hffs.set_repo_id(repo_id)
        for index in range(length):
            r = random.randint(1000000,9999999)
            for batch  in Batcher.BATCHES:
                datum = {}
                for layer_index in range(batch[0], batch[1]+1):
                    if layer_index in cls.hidden_states:
                        for k in cls.hidden_states[layer_index][index]: datum["{:0>2}".format(layer_index)+f"-{k}"] = cls.hidden_states[layer_index][index][k]
                cls.queue.put((Batcher.label(r,batch,MERGE_SIZE), datum))
            
        cls.reset_all()

class InternalsTracker(torch.nn.Module):
    all_datasets = {}
    def __init__(self, label:str, keep_last=0):
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
        if not len(cls.all_datasets): return
        if append and os.path.exists(filepath):
            old = load_file(filename=filepath)
            for k in cls.all_datasets: cls.all_datasets[k] += old.pop(k,0)
            for k in old: cls.all_datasets[k] = old[k]
        save_file(cls.all_datasets, filename=filepath)

        cls.reset_all()

    @classmethod
    def download_internals(cls):
        filepath = os.path.join(os.path.split(__file__)[0], 'data', 'internals.safetensors')
        remote   = UploadThread.hffs.rpath(os.path.basename(filepath))
        UploadThread.hffs.fs.get_file( rpath=remote, lpath=filepath )
        return { remote:filepath }

    @classmethod
    def upload_internals(cls):
        filepath = os.path.join(os.path.split(__file__)[0], 'data', 'internals.safetensors')
        remote   = UploadThread.hffs.rpath(os.path.basename(filepath))
        UploadThread.hffs.fs.put_file( lpath=filepath, rpath=remote)
        return { filepath:remote }

    @classmethod
    def inject_internals_tracker(cls,block:Union[DoubleStreamBlock, SingleStreamBlock, HiddenStateTracker], index:int):

        block = block.wrapped_module if isinstance(block, HiddenStateTracker) else block

        if hasattr(block, 'internals_tracker_added') and block.internals_tracker_added:
            return

        if isinstance(block, DoubleStreamBlock):
            block.img_mlp.insert(2, InternalsTracker("{:0>2}-img".format(index)))
            block.txt_mlp.insert(2, InternalsTracker("{:0>2}-txt".format(index)))
        elif isinstance(block, SingleStreamBlock):
            block.linear2 = torch.nn.Sequential(
                InternalsTracker("{:0>2}-x".format(index), keep_last=block.mlp_hidden_dim),
                block.linear2
            )

        block.internals_tracker_added = True
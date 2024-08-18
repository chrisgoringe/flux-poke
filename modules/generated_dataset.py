import os, torch
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath, shared
from safetensors.torch import load_file

class TheDataset:
    sources = None
    def __init__(self, dir, first_layer:int, split:str, thickness:int=1, train_frac=0.8):
        if os.path.isdir(local_dir:=filepath(dir)):
            self.sources = self.sources or [ os.path.join(local_dir,x) for x in os.listdir(local_dir) if x.endswith(".safetensors") ]
            self.load_file = load_file
        else:
            self.hffs = HFFS(repo_id=dir)
            self.sources = self.sources or self.hffs.get_entry_list()
            self.load_file = partial(self.hffs.load_file)
                                     
        split_at = int(train_frac*len(self.sources))
        if   split=='train': self.sources = self.sources[:split_at]
        elif split=='eval':  self.sources = self.sources[split_at:]
        self.first_layer = first_layer
        self.thickness   = thickness
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):

        input  = self.load_file(filename="/".join((self.sources[i], str(self.first_layer))))
        output = self.load_file(filename="/".join((self.sources[i], str(self.first_layer+self.thickness))))
        l1, l2 = "{:0>2}-".format(self.first_layer) , "{:0>2}-".format(self.first_layer+self.thickness)
        for k in input:
            if torch.isinf(input[k]).any():
                print("INF")
        for k in output:
            if torch.isinf(output[k]).any():
                print("INF")

        data = {}
        for k in ['img', 'txt', 'x', 'vec', 'pe']:
            if (x:=input.get( l1+k, None)) is not None:  data[k]        = x.squeeze(0)
            if (y:=output.get(l2+k, None)) is not None:  data[k+"_out"] = y.squeeze(0)

        return data
import os
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath
from safetensors.torch import load_file

class TheDataset:
    def __init__(self, dir, first_layer:int, split:str, thickness:int=1, train_frac=0.8, filter_hffs_cache=True):
        if os.path.isdir(local_dir:=filepath(dir)):
            self.sources = [ os.path.join(local_dir,x) for x in os.listdir(local_dir) if x.endswith(".safetensors") ]
            self.load_file = load_file
        else:
            self.hffs = HFFS(repository=dir)
            self.sources = self.hffs.get_file_list()
            if filter_hffs_cache:
                self.load_file = partial(self.hffs.load_file, filter=self.apply_filter)
            else:
                self.load_file = partial(self.hffs.load_file)
                                     
        split_at = int(train_frac*len(self.sources))
        if   split=='train': self.sources = self.sources[:split_at]
        elif split=='eval':  self.sources = self.sources[split_at:]
        self.layer = first_layer
        self.thickness = thickness
    
    def __len__(self): 
        return len(self.sources)
    
    def apply_filter(self, data):
        return { k:data[k] for k in data if (k.startswith(f"{self.layer}-") or k.startswith(f"{self.layer+self.thickness}-"))}

    def __getitem__(self, i):
        all_data = self.load_file(filename=self.sources[i])
        return {
            "img"     : all_data[f"{self.layer}-img"].squeeze(0),
            "txt"     : all_data[f"{self.layer}-txt"].squeeze(0),
            "vec"     : all_data[f"{self.layer}-vec"].squeeze(0),
            "pe"      : all_data[f"{self.layer}-pe"].squeeze(0),
            "img_out" : all_data[f"{self.layer+self.thickness}-img"].squeeze(0),
            "txt_out" : all_data[f"{self.layer+self.thickness}-txt"].squeeze(0),
        }
import os
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath
from safetensors.torch import load_file

class TheDataset:
    def __init__(self, dir, first_layer:int, split:str, thickness:int=1, train_frac=0.8, is_double=True):
        if os.path.isdir(local_dir:=filepath(dir)):
            self.sources = [ os.path.join(local_dir,x) for x in os.listdir(local_dir) if x.endswith(".safetensors") ]
            self.load_file = load_file
        else:
            self.hffs = HFFS(repo_id=dir)
            self.sources = self.hffs.get_entry_list()
            self.load_file = partial(self.hffs.load_file)
                                     
        split_at = int(train_frac*len(self.sources))
        if   split=='train': self.sources = self.sources[:split_at]
        elif split=='eval':  self.sources = self.sources[split_at:]
        self.layer = first_layer
        self.thickness = thickness
        self.is_double = is_double
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):
        input  = self.load_file(filename="/".join(self.sources[i], str(self.first_layer)))
        output = self.load_file(filename="/".join(self.sources[i], str(self.first_layer+self.thickness)))
        if self.is_double:
            return {
                "img"     : input["img"].squeeze(0),  "txt"     : input["txt"].squeeze(0),  
                "vec"     : input["vec"].squeeze(0),  "pe"      : input["pe"].squeeze(0),
                "img_out" : output["img"].squeeze(0), "txt_out" : output["txt"].squeeze(0),
            }
        else:
            return {
                "x"     : input["x"].squeeze(0),
                "vec"   : input["vec"].squeeze(0),  "pe" : input["pe"].squeeze(0),
                "x_out" : output["x"].squeeze(0),
            }
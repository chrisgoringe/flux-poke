import os, torch, random
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath, shared
from safetensors.torch import load_file

class TheDataset:
    sources = None
    shuffle = False

    @classmethod
    def set_dataset_source(cls, dir, shuffle=False, seed=0):
        if os.path.isdir(local_dir:=filepath(dir)):
            cls.sources = cls.sources or [ os.path.join(local_dir,x) for x in os.listdir(local_dir) if x.endswith(".safetensors") ]
            cls.load_file = load_file
        else:
            cls.hffs = HFFS(repo_id=dir)
            cls.sources = cls.sources or cls.hffs.get_entry_list()
            cls.load_file = partial(cls.hffs.load_file)
        if shuffle:
            if seed: random.seed(seed)
            random.shuffle(cls.sources)
        print("Dataset contains {:>5} folders".format(len(cls.sources)))

    def __init__(self, first_layer:int, split:str, thickness:int=1, train_frac=0.8):                
        split_at = int(train_frac*len(self.sources))
        if   split.lower()=='train': self.sources = self.sources[:split_at]
        elif split.lower()=='eval':  self.sources = self.sources[split_at:]
        elif split.lower()=='all':   pass
        else: assert False, f"Split must be 'train', 'eval', or 'all': got {split}"
        self.first_layer = first_layer
        self.thickness   = thickness
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):
        print(f"Using {self.sources[i]}")
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
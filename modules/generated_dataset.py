import os, random
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath
from safetensors.torch import load_file

EXCLUDE = [9091578,
]

class TheDataset:
    _sources = None
    shuffle = False

    @classmethod
    def set_dataset_source(cls, dir, shuffle=False, seed=0):
        if os.path.isdir(local_dir:=filepath(dir)):
            cls._sources = cls._sources or [ os.path.join(local_dir,x) for x in os.listdir(local_dir) if x.endswith(".safetensors") ]
            cls.load_file = load_file
        else:
            cls.hffs = HFFS(repo_id=dir)
            cls._sources = cls._sources or cls.hffs.get_entry_list()
            cls._sources = [x for x in cls._sources if int(x.split('/')[-1]) not in EXCLUDE]
            cls.load_file = partial(cls.hffs.load_file)
        if shuffle:
            if seed: random.seed(seed)
            random.shuffle(cls._sources)
        print("Complete dataset contains {:>5} examples".format(len(cls._sources)))

    def __init__(self, first_layer:int, split:str, thickness:int=1, train_frac=0.8, eval_frac=0.2):
        if   split.lower() == 'train':
            self.sources = self._sources[    : int(train_frac*len(self._sources)) ]
        elif split.lower() == 'eval':
            self.sources = self._sources[ int((1-eval_frac)*len(self._sources)) : ]
        elif split.lower() == 'all':
            self.sources = self._sources
        else: raise AttributeError(f"Split must be 'train', 'eval', or 'all': got {split}")
        print(f"Split {split} contains {len(self.sources)} examples")
        self.first_layer = first_layer
        self.thickness   = thickness
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):
        input  = self.load_file(filename="/".join((self.sources[i], str(self.first_layer))))
        output = self.load_file(filename="/".join((self.sources[i], str(self.first_layer+self.thickness))))
        l1, l2 = "{:0>2}-".format(self.first_layer) , "{:0>2}-".format(self.first_layer+self.thickness)

        data = {}
        for k in ['img', 'txt', 'x', 'vec', 'pe']:
            if (x:=input.get( l1+k, None)) is not None:  data[k]        = x.squeeze(0)
            if (y:=output.get(l2+k, None)) is not None:  data[k+"_out"] = y.squeeze(0)

        return data
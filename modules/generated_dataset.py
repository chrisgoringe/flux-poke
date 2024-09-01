import random
from functools import partial
from modules.hffs import HFFS
from warnings import warn

class _Dataset:
    _mapping = None

    def __init__(self,split,train_frac,eval_frac):
        if   split.lower() == 'train':
            self.mapping = self._mapping[    : int(train_frac*len(self._mapping)) ]
        elif split.lower() == 'eval':
            self.mapping = self._mapping[ int((1-eval_frac)*len(self._mapping)) : ]
        elif split.lower() == 'all':
            self.mapping = self._mapping
        else: raise AttributeError(f"Split must be 'train', 'eval', or 'all': got {split}")
        print(f"Split {split} contains {len(self.mapping)} examples")

    def __len__(self): 
        return len(self.mapping)
    
    def last_was(self):
        return ""
    

class RemoteDataset(_Dataset):
    shuffle = False
    @classmethod
    def set_dataset_source(cls, dir, shuffle=False, seed=0, exclusions:list[int]=[], validate=False):
        cls.hffs = HFFS(repo_id=dir)
        cls._mapping = cls._mapping or cls.hffs.get_entry_list(validate=validate)
        cls._mapping = [x for x in cls._mapping if int(x.split('/')[-1]) not in exclusions]

        if shuffle:
            if seed: random.seed(seed)
            random.shuffle(cls._mapping)
        print("Complete dataset contains {:>5} examples".format(len(cls._mapping)))

    def __init__(self, first_layer:int, split:str, thickness:int=1, train_frac=0.8, eval_frac=0.2, squeeze=True):
        super().__init__(split, train_frac, eval_frac)
        
        self.first_layer = first_layer
        self.thickness   = thickness
        self.last_source_was = None
        self.squeeze = squeeze

    def __getitem__(self, i):
        self.last_source_was = self.mapping[i]
        input  = self.hffs.load_file(filename="/".join((self.mapping[i], str(self.first_layer))))
        output = self.hffs.load_file(filename="/".join((self.mapping[i], str(self.first_layer+self.thickness))))
        l1, l2 = "{:0>2}-".format(self.first_layer) , "{:0>2}-".format(self.first_layer+self.thickness)

        data = {}
        for k in ['img', 'txt', 'x', 'vec', 'pe']:
            if (x:=input.get( l1+k, None)) is not None:  data[k]        = x.squeeze(0) if self.squeeze else x
            if (y:=output.get(l2+k, None)) is not None:  data[k+"_out"] = y.squeeze(0) if self.squeeze else y

        return data
    
    def last_was(self):
        return self.last_source_was

MERGE_SIZE = 20
class MergedBatchDataset(_Dataset):
    hffs:HFFS

    @classmethod
    def set_dataset_source(cls, dir, shuffle=False, seed=0, exclusions:list[int]=[], validate=False):
        if validate: warn("Validation not implemented for MergedBatchDataset (also not needed?)")
        cls.hffs = HFFS(repo_id=dir)
        sources = cls.hffs.get_entry_list(validate=False)
        sources = [x for x in sources if int(x.split('/')[-1]) not in exclusions]

        cls._mapping = []
        for source in sources:
            for i in range(MERGE_SIZE):
                cls._mapping.append( (source, i) )

        if shuffle:
            if seed: random.seed(seed)
            random.shuffle(cls._mapping)
        print("Complete dataset contains {:>5} examples".format(len(cls._mapping)))

    def __init__(self, split:str='all', train_frac=0.8, eval_frac=0.2):
        super().__init__(split, train_frac, eval_frac)
        self.last_source = None
        self.last_source_data = None
        
    KEY_MAP = [
        ("00-img", "img"),
        ("00-txt", "txt"),
        ("00-vec", "vec"),
        ("00-pe", "pe"),
        ("57-x", "x_out"),
    ]
    
    def __getitem__(self, i):
        source, entry = self.mapping[i]
        if source != self.last_source:
            self.last_source = source
            self.last_source_data = self.hffs.load_file(filename="/".join((source, "all_{:0>2}.safetensors".format(MERGE_SIZE))))
        self.last_entry = entry
        
        label = lambda key : f"{entry:0>2}_{key}"
        datum = { dkey:self.last_source_data[label(key)] for key, dkey in self.KEY_MAP }
        return datum

    def last_was(self):
        return (self.last_source, self.last_entry)
import os, torch, random
from functools import partial
from modules.hffs import HFFS
from modules.utils import filepath, shared
from safetensors.torch import load_file
BAD = [
    1018720,
1135114,
1158742,
1199123,
1233515,
1265963,
1298962,
1310151,
1332861,
1340568,
1355537,
1393368,
1699386,
1731737,
1808640,
2054713,
2111704,
2436806,
2637835,
2699408,
2789006,
2806252,
3008511,
3286407,
3337307,
3364960,
3413469,
3443117,
3448117,
3659893,
3664764,
3674352,
3721812,
3899173,
4059699,
4102300,
4125701,
4139995,
4166979,
4218165,
4235931,
4421116,
4464822,
4553444,
4554065,
4597040,
4649668,
4654406,
4999673,
5061557,
5139757,
5161298,
5270527,
5298499,
5383785,
5475709,
5486208,
5556656,
5676993,
5783702,
5873525,
6040602,
6080422,
6116182,
6172036,
6214934,
6339076,
6437492,
6441834,
6512756,
6642743,
6646304,
6852597,
6908003,
6956457,
6979255,
7029558,
7078819,
7105710,
7219754,
7373207,
7452979,
7507585,
7797250,
7910159,
7961146,
8013900,
8260976,
8261631,
8543452,
8578703,
8587848,
8819356,
8975260,
9101529,
9136274,
9203861,
9287865,
9293340,
9567785,
9711542,
9864127,
9906778,
]

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
            cls.sources = [x for x in cls.sources if int(x.split('/')[-1]) not in BAD]
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
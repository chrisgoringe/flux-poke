import logging, time, yaml, os, shutil
from safetensors.torch import load_file
from functools import partial

filepath = partial(os.path.join,os.path.split(__file__)[0],"..")

class SingletonAddin:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class Log(SingletonAddin):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start = time.monotonic()

    def info(self, msg):
        self.logger.info("{:>8.1f}s ".format(time.monotonic()-self.start)+msg)

log = Log.instance().info

def range_from_string(s):
    if s is None: return []
    if isinstance(s,int): return [s,]
    if not '-' in s: return [int(s),]
    a,b = (int(x.strip()) for x in s.split('-'))
    return [x for x in range(a,b+1)]

def preserve_existing_file(filename):
    if os.path.exists(filename):
        name, ext = os.path.splitext(filename)
        copy_number = 1
        def nth_copy(x): return f"{name}_({x}){ext}"
        while os.path.exists( nth_copy(copy_number) ): copy_number = copy_number + 1
        shutil.move(filename, nth_copy(copy_number) )
        log(f"Renamed {filename} to {nth_copy(copy_number)}")

class Shared(SingletonAddin):
    def __init__(self):
        self.max_layer = 56
        self.last_double_layer = 18
        self.layer_stats = [{} for _ in range(57)]

    def load(self,args):
        self.sd        = load_file(args.model)
        self.internals = load_file(filepath(args.internals))

    @property
    def layer_stats_yaml(self):
        return yaml.dump( { "layer{:0>2}".format(i):layer_stats for i, layer_stats in enumerate(self.layer_stats) if layer_stats} )
    
    def save_stats(self, filename):
        preserve_existing_file(filename)
        with open(filename, 'w') as f: 
            print(shared.layer_stats_yaml, file=f)
            log(f"Saved stats in {filename}")

class Batcher:
    BATCHES = (( 0, 6), ( 7,13), (14,18), (19,25), (26,31), (32,37), (38,43), (44,50), (51,57))
    @classmethod
    def filename(cls, base_name, layer_index):
        for a,b in cls.BATCHES:
            if layer_index>=a and layer_index<=b:
                return "{:0>7}/{:0>2}-{:0>2}.safetensors".format(base_name,a,b)
            
    @classmethod
    def label(cls, base_name, batch): return "{:0>7}/{:0>2}-{:0>2}.safetensors".format(base_name, *batch)
    
shared = Shared.instance()

import logging, time, yaml, os, shutil, json
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

def load_config(config_filepath):
    if os.path.splitext(config_filepath)[1]==".yaml":
        with open(config_filepath, 'r') as f: return yaml.safe_load(f)
    else:
        with open(config_filepath, 'r') as f: return json.load(f)

def int_list_from_string(s):
    rnge = []
    for section in (x.strip() for x in str(s or "").split(',')):
        if section:
            a,b = (int(x.strip()) for x in section.split('-')) if '-' in section else (int(section), int(section))
            for i in range(a,b+1): rnge.append(i)
    return rnge

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
        self._sd        = None
        self._internals = None
        self.args       = None

    @property
    def sd(self):
        if isinstance(self._sd,str): self._sd = load_file(self._sd)
        return self._sd

    @property
    def internals(self):
        if isinstance(self._internals, str): self._internals = load_file(self._internals)
        return self._internals

    def load(self,args):
        self._sd        = args.model
        self._internals = filepath(args.internals)

    @property
    def layer_stats_yaml(self):
        return yaml.dump( { "layer{:0>2}".format(i):layer_stats for i, layer_stats in enumerate(self.layer_stats) if layer_stats} )

    @property
    def layer_stats_json(self):
        return json.dumps( { "layer{:0>2}".format(i):layer_stats for i, layer_stats in enumerate(self.layer_stats) if layer_stats}, indent=2 )
        
    def save_stats(self, filename):
        format_yaml = (os.path.splitext(filename)[1]==".yaml")
        preserve_existing_file(filename)
        with open(filename, 'w') as f: 
            print(shared.layer_stats_yaml if format_yaml else shared.layer_stats_json, file=f)
            log(f"Saved stats in {filename}")

def is_double(layer_number): return (layer_number <= shared.last_double_layer)

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

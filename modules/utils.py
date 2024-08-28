import logging, time, yaml, os, shutil, json
from safetensors.torch import load_file
from functools import partial
from warnings import warn
import torch
from typing import Iterable

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

def layer_iteratable_from_string(s) -> Iterable[int]:
    if isinstance(s, int): return [s,]
    if s.lower()=='all':    return range(shared.last_layer+1)
    if s.lower()=='double': return range(shared.first_double_layer, shared.last_double_layer+1)
    if s.lower()=='single': return range(shared.first_single_layer, shared.last_single_layer+1)

    def parse():
        for section in (x.strip() for x in str(s or "").split(',')):
            if section:
                a,b = (int(x.strip()) for x in section.split('-')) if '-' in section else (int(section), int(section))
                for i in range(a,b+1): yield i
    return parse()

def preserve_existing_file(filename):
    if os.path.exists(filename):
        name, ext = os.path.splitext(filename)
        copy_number = 1
        def nth_copy(x): return f"{name}_({x}){ext}"
        while os.path.exists( nth_copy(copy_number) ): copy_number = copy_number + 1
        shutil.move(filename, nth_copy(copy_number) )
        log(f"Renamed {filename} to {nth_copy(copy_number)}")

def prefix(layer_index):
    if is_double(layer_index):
        return f"double_blocks.{layer_index}."
    else:
        return f"single_blocks.{layer_index-shared.first_single_layer}."
    
class FluxFacts:
    last_layer = 56
    first_double_layer = 0
    last_double_layer  = 18
    first_single_layer = 19
    last_single_layer  = 56    
    bits_at_bf16       = 190422221824
    
class Shared(SingletonAddin, FluxFacts):
    def __init__(self):
        self.layer_stats = [{} for _ in range(self.last_layer+1)]
        self._sd:dict[str,torch.Tensor] = None
        self._internals = None
        self.args       = None
        self._layerssd:dict[str,dict[str,torch.Tensor]]  = None
        self.masks:dict[str,list[bool]] = {}

    @property
    def sd(self):
        if isinstance(self._sd,str): self._sd = load_file(self._sd)
        return self._sd
    
    def store_mask(self, mask, layer_index, subtype):
        index = f"{layer_index:0>2}.{subtype}"
        self.masks[index] = mask

    def get_masks(self, layer_index):
        return { k[3:]:v for k,v in self.masks.items() if k.startswith("{:0>2}".format(layer_index)) }
    
    def layer_sd(self, layer_index):
        if self._layerssd is None: self.split_sd()
        return self._layerssd[prefix(layer_index)]
    
    def drop_layer(self, layer_index):
        for k in self._layerssd.pop(prefix(layer_index)): self._sd.pop(prefix(layer_index)+k)
        
    def split_sd(self):
        self._layerssd = { prefix(x):{} for x in range(self.last_layer+1) }
        for k in self.sd:
            for pf in self._layerssd:
                if k.startswith(pf): 
                    self._layerssd[pf][k[len(pf):]] = self.sd[k]
                    break

    @property
    def internals(self):
        if isinstance(self._internals, str): self._internals = load_file(self._internals)
        return self._internals

    def set_shared_filepaths(self,args):
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
    _BATCHED   = (( 0, 6), ( 7,13), (14,18), (19,25), (26,31), (32,37), (38,43), (44,50), (51,57))
    _FULL      = (( 0,57), )
    BATCHES    = _BATCHED
    all_in_one = False

    @classmethod
    def set_mode(cls, all_in_one):
        cls.all_in_one = all_in_one
        if all_in_one: cls.BATCHES = cls._FULL
        else:          cls.BATCHES = cls._BATCHED

    @classmethod
    def filename(cls, base_name, layer_index):
        for a,b in cls.BATCHES:
            if layer_index>=a and layer_index<=b:
                return "{:0>7}/{:0>2}-{:0>2}.safetensors".format(base_name,a,b)
            
    @classmethod
    def label(cls, base_name, batch, number=1): 
        if cls.all_in_one:
            return "{:0>7}/all_{:0>2}.safetensors".format(base_name, number)
        else:
            return "{:0>7}/{:0>2}-{:0>2}.safetensors".format(base_name, *batch)


    
shared = Shared.instance()

import os, random, re, time
from functools import partial 
from safetensors.torch import load_file
import torch
from .modules.trackers import UploadThread

filepath = partial(os.path.join,os.path.split(__file__)[0])

class SeedContext():
    """
    Context Manager to allow one or more random numbers to be generated, optionally using a specified seed, 
    without changing the random number sequence for other code.
    """
    def __init__(self, seed=None):
        self.seed = seed
    def __enter__(self):
        self.state = random.getstate()
        if self.seed:
            random.seed(self.seed)
    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.state)

class QPause:
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "latent":("LATENT",{}), "limit":("INT",{"default":10})}}

    def func(self, latent, limit):
        while UploadThread.instance().queue.qsize() > limit:
            time.sleep(10)
        return (latent,)

class Prompts:
    RETURN_TYPES = ("STRING","STRING","STRING")
    RETURN_NAMES = ("prompt","index","seed")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "index": ("INT", {"default": -1 }), 
        "seed": ("INT", {"default": 42 }),
        "reload":(["no","yes"],), 
        "filename":("STRING", {"default":"prompts_dataset"}),
    } }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def __init__(self):
        self.prompts = None

    def load(self, filename):
        if os.path.isdir(filepath(filename)):
            import datasets
            self.prompts = [p for p in datasets.Dataset.load_from_disk(filepath(filename))['prompt']]
        else:
            with open(filepath(filename), 'r', encoding='UTF-8') as f: self.prompts = f.readlines()

    def func(self, index, seed, reload, filename):
        if reload=='yes' or self.prompts is None: 
            self.load(filename)
            self.last_regex = ".*"
        print(f"Prompts has {len(self.prompts)} entries")
        if index==-1: 
            with SeedContext(seed): index = random.randrange(0, len(self.prompts))
        return (self.prompts[index % len(self.prompts)], str(index), str(seed))

class Counter:
    RETURN_TYPES = ("INT","STRING")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "max": ("INT", {"default": 1 }), "restart":(["no","yes"],)} }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")
    
    def __init__(self):
        self.n = -1

    def func(self, max, restart):
        if restart=='yes': self.n = -1
        self.n = (self.n+1) % max
        return (self.n,str(self.n))

class RandomInt:
    RETURN_TYPES = ("INT",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": {
        "seed":    ("INT", {"default": 42 }), 
        "minimum": ("INT", {"default": 15 }), 
        "maximum": ("INT", {"default": 30 }),
    }}

    def func(self,seed, minimum, maximum):
        with SeedContext(seed):
            return (random.randint(minimum, maximum),)

class RandomSize:
    SIZES = [
        (1024,1024),
        (896,1152),
        (864,1536),
        (832,1216),
        (768,1024),
        (768,1280),
        (768,1344),
        (720,1280),
        (624,800),
        (576,1024),
    ]
    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("W","H")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "seed": ("INT", {"default": 42 }), "flip": ("FLOAT", {"default":0.5, "min":0.0, "max":1.0, "step":0.01 }), } }

    def func(self,seed, flip):
        with SeedContext(seed):
            w, h = random.choice(self.SIZES)
            return ( h, w ) if (random.random()<flip) else ( w,h )
        
class CommonSizes(RandomSize):
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "size": ([ f"{x}x{y}" for (x,y) in s.SIZES ] + [ f"{y}x{x}" for (x,y) in s.SIZES if x!=y ],{}), } }
    
    def func(self,size:str):
        x,y = (int(x) for x in size.split('x'))
        return (x,y)
'''
class FluxSuperScheduler:
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    def simple_scheduler(self):
        sigs = [1.0, 0.99991, 0.0.0,]
        return torch.FloatTensor(sigs, device="cpu")
    
    def func(self, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0: return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        return (self.simple_scheduler(total_steps, steps),)
'''
class FluxSimpleScheduler:
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }}
    
    def __init__(self):
        self.model_sampling = load_file(filepath("data","flux_sigmas.safetensors"))['sigmas']

    def simple_scheduler(self, total_steps, steps):
        sigs = [float(self.model_sampling[-(1 + int(x * len(self.model_sampling) / total_steps))]) for x in range(total_steps)] + [0.0,]
        return torch.FloatTensor(sigs, device="cpu")[-(steps + 1):]
    
    def func(self, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0: return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        return (self.simple_scheduler(total_steps, steps),)
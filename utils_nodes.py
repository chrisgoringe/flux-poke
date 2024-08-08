import os, random, re
from functools import partial 
from safetensors.torch import load_file
import torch

filepath = partial(os.path.join,os.path.split(__file__)[0])

class Prompts:
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt","index")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "index": ("INT", {"default": 0 }), 
        "reload":(["no","yes"],), 
        "filename":("STRING", {"default":"prompts.txt"}),
        "regex":("STRING", {"default":".*"},)
    } }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def __init__(self):
        self.prompts = None
        self.last_regex = ".*"

    def load(self, filename):
        with open(filepath(filename), 'r', encoding='UTF-8') as f: self.prompts = f.readlines()

    def filter(self, regex_string):
        regex = re.compile(regex_string)
        self.prompts = [ x for x in self.prompts if regex.match(x) ]

    def func(self, index, reload, filename, regex):
        if reload=='yes' or self.prompts is None: 
            self.load(filename)
            self.last_regex = ".*"
        if regex != self.last_regex:
            self.filter(regex)
            self.last_regex = regex
        print(f"Prompts has {len(self.prompts)} entries")
        if index==-1: index = random.randrange(0, len(self.prompts))
        return (self.prompts[index % len(self.prompts)], str(index))

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
        self.model_sampling = load_file(filepath("flux_sigmas.safetensors"))['sigmas']

    def simple_scheduler(self, total_steps, steps):
        sigs = [float(self.model_sampling[-(1 + int(x * len(self.model_sampling) / total_steps))]) for x in range(total_steps)] + [0.0,]
        return torch.FloatTensor(sigs, device="cpu")[-(steps + 1):]
    
    def func(self, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0: return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        return (self.simple_scheduler(total_steps, steps),)
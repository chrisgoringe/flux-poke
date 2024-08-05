import os, random
ROOT = os.path.split(__file__)[0]

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
    } }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def __init__(self):
        self.prompts = None

    def load(self, filename):
        with open(os.path.join(ROOT, filename), 'r', encoding='UTF-8') as f: self.prompts = f.readlines()

    def func(self, index, reload, filename):
        if reload=='yes' or self.prompts is None: self.load(filename)
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
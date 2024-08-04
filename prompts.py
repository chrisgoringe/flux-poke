
class Prompts:
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("prompt","index")
    FUNCTION = "func"
    CATEGORY = "flux_watcher"
    @classmethod
    def INPUT_TYPES(s): return { "required": { "index": ("INT", {"default": 0 }), "reload":(["no","yes"],)} }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def __init__(self):
        self.load()

    def load(self):
        with open("prompts.txt", 'r') as f: self.prompts = f.readlines()

    def func(self, index, reload):
        if reload=='yes': self.load()
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
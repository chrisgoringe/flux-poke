import torch
import asyncio

def setup_prepares(module:torch.nn.Module):
    preparable = [m for m in module.modules() if hasattr(m,'prepare')]
    def prepare(*args):
        for p in preparable: asyncio.run(p.prepare()) 
    module.register_forward_pre_hook(prepare)

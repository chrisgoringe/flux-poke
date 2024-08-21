import torch
import asyncio
from .arguments import args

def async_run_prepares(module:torch.nn.Module):
    if args.run_asyncs:
        preparable = [m for m in module.modules() if hasattr(m,'prepare')]
        async def run_prepares():
            for p in preparable: p.prepare()
        asyncio.run(run_prepares)

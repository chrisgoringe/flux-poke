import torch
from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock
from comfy.model_patcher import ModelPatcher

class OnDemandLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, cpu_bfloat16:bool):
        super().__init__()
        if cpu_bfloat16: linear.to(torch.bfloat16)
        self.wrapped = linear

    def _apply(self, fn, recurse=True): pass

    def forward(self, x:torch.Tensor):
        weight = self.wrapped.weight.cuda()
        bias   = self.wrapped.bias.cuda() if self.wrapped.bias is not None else None
        with torch.autocast("cuda"):
            return torch.nn.functional.linear(x, weight, bias)

def lock_SingleStreamBlock_to_cpu(module:SingleStreamBlock, cpu_bfloat16):
    module.linear1 = OnDemandLinear(module.linear1, cpu_bfloat16)
    module.linear2 = OnDemandLinear(module.linear2, cpu_bfloat16)
    # attention to add

def lock_DoubleStreamBlock_to_cpu(module:DoubleStreamBlock, cpu_bfloat16):
    module.txt_mlp = torch.nn.Sequential(OnDemandLinear(module.txt_mlp[0], cpu_bfloat16), module.txt_mlp[1], 
                                         OnDemandLinear(module.txt_mlp[2], cpu_bfloat16))
    module.img_mlp = torch.nn.Sequential(OnDemandLinear(module.img_mlp[0], cpu_bfloat16), module.img_mlp[1], 
                                         OnDemandLinear(module.img_mlp[2], cpu_bfloat16))
    # attention to add

def split_model(model:ModelPatcher, number_of_single_blocks:int, number_of_double_blocks:int, cpu_bfloat16:bool):
    '''
    Run the SingleStreamBlock stack on the CPU
    '''
    diffusion_model:Flux = model.model.diffusion_model
    for i in range(min(number_of_single_blocks, len(diffusion_model.single_blocks))):
        lock_SingleStreamBlock_to_cpu(diffusion_model.single_blocks[i], cpu_bfloat16)
    for i in range(min(number_of_double_blocks, len(diffusion_model.double_blocks))):
        lock_DoubleStreamBlock_to_cpu(diffusion_model.double_blocks[i], cpu_bfloat16)
    return model
    
class CPUOffLoad:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_watcher"

    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "model": ("MODEL",{}),
        "double_blocks_on_cpu": ("INT",{"default":0, "min":0, "max":19}),
        "single_blocks_on_cpu": ("INT",{"default":0, "min":0, "max":38}),
        "cpu_bfloat16":         (["yes","no"],{})
    } }

    def func(self, model, double_blocks_on_cpu, single_blocks_on_cpu, cpu_bfloat16): 
        m = split_model(model.clone(), number_of_single_blocks=single_blocks_on_cpu, 
                        number_of_double_blocks=double_blocks_on_cpu, cpu_bfloat16=(cpu_bfloat16=='yes'))
        
        fraction_pinned   = ( 2*double_blocks_on_cpu + single_blocks_on_cpu ) / 57 
        fraction_possible = 0.5 # what fraction of the whole thing could we possibly have pinned? Work out better!

        m.model.memory_usage_factor *= (1 - fraction_possible*fraction_pinned)
        return (m,)

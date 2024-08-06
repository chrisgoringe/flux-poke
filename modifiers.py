from comfy.ldm.flux.layers import DoubleStreamBlock
import torch

def prune(t:torch.Tensor, mask):
    t, transpose = (t.T, True) if len(t.shape)==2 and t.shape[1] == len(mask) else (t, False)
    if t.shape[0] != len(mask): return t
    t = torch.stack( list(t[i,...] for i,m in enumerate(mask) if m ) )
    return t.T if transpose else t

class Info:
    def __init__(self,mlp):
        mlp0:torch.nn.Linear = mlp[0]
        self.clazz             = mlp0.__class__
        self.dtype             = mlp0.weight.dtype
        self.device            = mlp0.weight.device
        self.hidden_size       = mlp0.in_features
        self.intermediate_size = mlp0.out_features
        
def new_mlp(old_mlp, mask):
    assert len(mask)==old_mlp[0].weight.shape[0]
    if all(mask): return old_mlp

    info = Info(old_mlp)
    hidden_dim  = sum(mask)

    mlp = torch.nn.Sequential(
        info.clazz(info.hidden_size, hidden_dim, bias=True, dtype=info.dtype, device=info.device),
        torch.nn.GELU(approximate="tanh"),
        info.clazz(hidden_dim, info.hidden_size, bias=True, dtype=info.dtype, device=info.device),        
    )
    old_sd = old_mlp.state_dict()
    sd = { k:prune(old_sd[k],mask) for k in old_sd }
    mlp.load_state_dict(sd)
    return mlp

def slice_double_block(block:DoubleStreamBlock, img_mask:list[bool], txt_mask:list[bool]):
    block.img_mlp = new_mlp(block.img_mlp, img_mask)
    block.txt_mlp = new_mlp(block.txt_mlp, txt_mask)

def replace_double_block_mlps(block:DoubleStreamBlock, data:dict[str,torch.Tensor]):
    img_lines, txt_lines = data['img_mlp.0.weight'].shape[0], data['txt_mlp.0.weight'].shape[0]
    info = Info(block.txt_mlp)
    block.img_mlp = torch.nn.Sequential(
        info.clazz(info.hidden_size, img_lines, bias=True, dtype=info.dtype, device=info.device),
        torch.nn.GELU(approximate="tanh"),
        info.clazz(img_lines, info.hidden_size, bias=True, dtype=info.dtype, device=info.device),        
    )
    block.txt_mlp = torch.nn.Sequential(
        info.clazz(info.hidden_size, txt_lines, bias=True, dtype=info.dtype, device=info.device),
        torch.nn.GELU(approximate="tanh"),
        info.clazz(txt_lines, info.hidden_size, bias=True, dtype=info.dtype, device=info.device),        
    )
    block.load_state_dict({ k:data[k].to(info.device, info.dtype) for k in data })
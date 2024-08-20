from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch, random
from warnings import warn

def prune(t:torch.Tensor, mask):
    t, transpose = (t.T, True) if len(t.shape)==2 and t.shape[1] == len(mask) else (t, False)
    if t.shape[0] != len(mask): return t
    t = torch.stack( list(t[i,...] for i,m in enumerate(mask) if m ) )
    return t.T if transpose else t

class Info:
    def __init__(self,mlp_ssb):
        linear                 = mlp_ssb.linear1 if isinstance(mlp_ssb, SingleStreamBlock) else mlp_ssb[0]
        self.clazz             = linear.__class__
        self.dtype             = linear.weight.dtype
        self.device            = linear.weight.device
        self.hidden_size       = linear.in_features
        self.intermediate_size = linear.out_features
        
def new_mlp(old_mlp, mask):
    if all(mask): return old_mlp
    assert len(mask)==old_mlp[0].weight.shape[0]
    
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

def get_mask(data, remove_below=None, remove_count=None): 

    if remove_count:
        if remove_below: warn("remove_below and remove_count both set: using remove_count")
        threshold = sorted(data)[remove_count]
        mask = [ d>=threshold for d in data ]
        matches_to_remove = (remove_count - sum(not x for x in mask))
        if matches_to_remove:
            matching_indices = [ index for index in range(len(mask)) if data[index]==threshold ]
            for index in random.sample(matching_indices, matches_to_remove): mask[index] = False
        return (mask, int(threshold))
    elif remove_below:  
        return ([ d>=remove_below for d in data ], remove_below)
    else:
        return ([True]*len(data), -1)

def slice_single_block(block:SingleStreamBlock, mask:list[bool]):
    info = Info(block)

    def new_linear(old_linear, mask, out=False):
        if out:
            new = info.clazz(in_features=sum(mask), out_features=info.hidden_size, dtype=info.dtype, device=info.device)
        else:
            new = info.clazz(in_features=info.hidden_size, out_features=sum(mask), dtype=info.dtype, device=info.device)
        old_sd = old_linear.state_dict()
        new_sd = { k:prune(old_sd[k],mask) for k in old_sd }
        new.load_state_dict( new_sd )
        return new

    block.linear1 = new_linear(block.linear1, mask = [True] * 3 * block.hidden_size + mask)
    block.linear2 = new_linear(block.linear2, mask = [True] * block.hidden_size + mask, out=True)
    block.mlp_hidden_dim = sum(mask)

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
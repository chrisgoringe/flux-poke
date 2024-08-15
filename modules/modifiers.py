from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch, random

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

def get_mask(data, remove_below, remove_count, remove_nothing, return_threshold = False): 
    if remove_nothing: return ([True]*len(data), -1) if return_threshold else [True]*len(data)
    remove_below, remove_count = (remove_below, None) if remove_below is not None else (sorted(data)[remove_count], remove_count)
    mask = [ d>=remove_below for d in data ] 
    # because of equalities, we may well not have removed enough. Take more out at random from those which equal remove_below
    if remove_count is not None and (still_to_remove := remove_count - sum(not x for x in mask)):
        jumbled = [x for x in range(len(data))]
        random.shuffle(jumbled)
        for pointer in jumbled:
            if still_to_remove>0 and data[pointer]==remove_below:
                mask[pointer] = False
                still_to_remove -= 1        
    return (mask, remove_below) if return_threshold else mask

def slice_single_block(block:SingleStreamBlock, mask:list[bool]):
    mask = [True] * 3 * block.hidden_size + mask
    info = Info(block)

    def new_linear(old_linear):
        new = info.clazz(in_features=info.hidden_size, out_features=sum(mask), dtype=info.dtype, device=info.device)
        old_sd = old_linear.state_dict()
        new_sd = { k:prune(old_sd[k],mask) for k in old_sd }
        new.load_state_dict( new_sd )
        return new

    block.linear1 = new_linear(block.linear1)
    block.linear2 = new_linear(block.linear2)

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
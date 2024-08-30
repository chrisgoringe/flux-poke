import add_paths
from modules.arguments import args, filepath
from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset
from modules.utils import Batcher, shared, is_double, load_config
from modules.casting import cast_layer_stack
from modules.pruning import prune_layer_stack
import torch
from tqdm import tqdm, trange
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import time, os

def new_layer(n) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    if is_double(n):
        return DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    else:
        return SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)

def load_single_layer(layer_number:int, remove_from_sd=True) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    layer_sd = shared.layer_sd(layer_number)
    if remove_from_sd: shared.drop_layer(layer_number)
    layer = new_layer(layer_number)
    layer.load_state_dict(layer_sd)
    return layer

def compute_loss(model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], autocast:bool) -> float:
    img, txt, vec, pe, x_out = inputs['img'].cuda(), inputs['txt'].cuda(), inputs['vec'].cuda(), inputs['pe'].cuda(), inputs['x_out'].cuda()
    x = None
    loss_fn = torch.nn.MSELoss()

    with torch.autocast("cuda", enabled=autocast):
        for i, layer in enumerate(model): 
            if isinstance(layer, DoubleStreamBlock): 
                img, txt = layer( img, txt, vec, pe ) 
            else:
                if x is None: x = torch.cat((txt, img), dim=1)
                x = layer( x, vec, pe )

    return(float(loss_fn(x, x_out)))

def setup():
    HFFS_Cache.set_cache_directory(args.cache_dir)
    shared.set_shared_filepaths(args=args)
    Batcher.set_mode(all_in_one=True)
    MergedBatchDataset.set_dataset_source(dir=args.hs_dir)
    
def create_dataset():
    return MergedBatchDataset(split='eval', eval_frac=args.eval_frac)

def load_layer_stack():
    print("Load model...")
    layer_stack = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    return layer_stack

def modify_layer_stack(layer_stack:torch.nn.Sequential, cast_config, prune_config):
    if cast_config:
        cast_layer_stack(layer_stack, cast_config=cast_config, 
                            stack_starts_at_layer=0, default_cast=args.default_cast, 
                            verbose=args.verbose, autocast=args.autocast)
    if prune_config:
        prune_layer_stack(layer_stack, prune_config=prune_config, model_first_layer=0, verbose=args.verbose)
    
def clone_layer_sd(layer_stack:torch.nn.Sequential, layer_number) -> dict[str,torch.Tensor]:
    sd:dict[str, torch.Tensor] = layer_stack[layer_number].state_dict()
    return { k:sd[k].clone() for k in sd }

def restore_layer(layer_stack:torch.nn.Sequential, sd, layer_number) -> torch.nn.Sequential:
    the_layer = new_layer(layer_number)
    the_layer.load_state_dict( sd )
    layer_stack = torch.nn.Sequential( *[m if i!=layer_number else the_layer for i, m in enumerate(layer_stack)] )
    return layer_stack

def evaluate(layer_stack, dataset, autocast:bool):
    layer_stack.cuda()
    with torch.no_grad():
        r = [ compute_loss(layer_stack, entry, autocast) for entry in tqdm(dataset) ]
    layer_stack.cpu()
    return r
    
def main():
    setup()
    the_data      = create_dataset()
    layer_stack   = load_layer_stack()

    BLOCKS = ['linear']
    CASTS = ['Q8_0', 'Q5_1', 'Q4_1']
    LAYERS = range(19, 57)

    outfile = os.path.join(args.save_dir, args.results_file)

    if not os.path.exists(outfile):
        with open(outfile, 'a') as output:
            print ("layer, block, cast, prune, loss, average full stack time", file=output, flush=True)

    for block in BLOCKS:
        for cast in CASTS:
            for layer in LAYERS:
                if (False):
                    pass
                else:
                    saved_layer_sd = clone_layer_sd(layer_stack, layer)
                    modify_layer_stack( layer_stack, 
                                        cast_config  = { 'casts': [{'layers': layer, 'blocks': block, 'castto': cast}] },
                                        prune_config = None )
                    start_time = time.monotonic()
                    mses = evaluate(layer_stack, the_data, args.autocast)
                    time_taken = (time.monotonic() - start_time)/len(the_data)
                    mean = sum(mses)/len(mses)
                    with open(outfile, 'a') as output:
                        print(f"{layer:>2},{block},{cast},0,{mean:>10.5},{time_taken:>10.5}", file=output, flush=True)
                    layer_stack = restore_layer(layer_stack, saved_layer_sd, layer)

if __name__=='__main__': 
    main()
    shared.save_stats(filepath(args.save_dir,args.stats_file))
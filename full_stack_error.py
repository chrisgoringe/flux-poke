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
from functools import partial

def clone_layer_sd(layer_stack:torch.nn.Sequential, layer_number) -> dict[str,torch.Tensor]:
    sd:dict[str, torch.Tensor] = layer_stack[layer_number].state_dict()
    return { k:sd[k].clone() for k in sd }

def restore_layer(layer_stack:torch.nn.Sequential, sd, layer_number) -> torch.nn.Sequential:
    the_layer = new_layer(layer_number)
    the_layer.load_state_dict( sd )
    layer_stack = torch.nn.Sequential( *[m if i!=layer_number else the_layer for i, m in enumerate(layer_stack)] )
    return layer_stack

class Result:
    def __init__(self, label):
        self.label:str          = label
        self.time:float         = None
        self.losses:list[float] = None
        self.loss:float         = None

class Job:
    def __init__(self, label:str, config:dict, preserve_layers:list[int]=[], prerun:callable=None, postrun:callable=None):
        self.config = config
        self.preserve_layers = preserve_layers
        self.prerun = prerun
        self.postrun = postrun
        self.result = Result(label)

    def execute(self, layer_stack:torch.nn.Sequential, the_data) -> Result:
        if self.prerun: self.prerun()

        saved_layer_sds = {layer_index:clone_layer_sd(layer_stack, layer_index) for layer_index in self.preserve_layers}
        modify_layer_stack( layer_stack, 
                            cast_config  = self.config if 'casts'  in self.config else None,
                            prune_config = self.config if 'prunes' in self.config else None )
        
        layer_stack.cuda()
        start_time = time.monotonic()
        losses = evaluate(layer_stack, the_data)
        self.result.time   = time.monotonic() - start_time
        self.result.losses = losses
        self.result.loss   = sum(losses) / len(losses)
        layer_stack.cpu()

        for layer_index, saved_layer_sd in saved_layer_sds.values():
            layer_stack = restore_layer(layer_stack, sd=saved_layer_sd, layer_number=layer_index)

        if self.postrun: self.postrun()
        return self.result

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

    loss = float(loss_fn(x, x_out))
    print(f"Autocast {autocast}, loss {loss}")
    return(loss)

def setup():
    HFFS_Cache.set_cache_directory(args.cache_dir)
    shared.set_shared_filepaths(args=args)
    Batcher.set_mode(all_in_one=True)
    MergedBatchDataset.set_dataset_source(dir=args.hs_dir)
    
def create_dataset():
    return MergedBatchDataset(split='eval', eval_frac=args.eval_frac)

def load_layer_stack():
    print("Loading model")
    layer_stack = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    return layer_stack

def modify_layer_stack(layer_stack:torch.nn.Sequential, cast_config, prune_config):
    if cast_config:
        cast_layer_stack(layer_stack, cast_config=cast_config, 
                            stack_starts_at_layer=0, default_cast=args.default_cast, 
                            verbose=args.verbose, autocast=args.autocast)
    if prune_config:
        prune_layer_stack(layer_stack, prune_config=prune_config, model_first_layer=0, verbose=args.verbose)
    
def evaluate(layer_stack, dataset):
    with torch.no_grad():
        return [ compute_loss(layer_stack, entry, args.autocast) for entry in tqdm(dataset) ]

'''
def get_jobs_list():
    BLOCKS = ['linear']
    CASTS = ['Q8_0', 'Q5_1', 'Q4_1']
    LAYERS = range(19, 57)
    jobs = []
    for block in BLOCKS:
        for cast in CASTS:
            for layer in LAYERS:
                if (cast=="Q8_0"):
                    pass
                else:
                    config = { 'casts': [{'layers': layer, 'blocks': block, 'castto': cast}] }
                    label = f"{layer},{block},{cast}"
                    jobs.append((label, config, [layer,]))
    return jobs'''


def get_jobs_list() -> list[Job]:
    jobs = []

    def set_autocast(x:bool): args.autocast = x

    jobs.append( Job("autocast", {}, [], prerun=lambda : partial(set_autocast, True)) )
    jobs.append( Job("noautocast", {}, [], prerun=lambda : partial(set_autocast, False)) )

    return jobs

'''
def get_jobs_list() -> list[Job]:
    for first_layer in [4,]:
        for second_layer in [5,9,14]:
            for first_block in ['img', 'txt']:
                for second_block in ['img', 'txt']:
                    config = { 'casts': [{'layers': first_layer, 'blocks': first_block, 'castto': 'Q4_1'},
                                        {'layers': second_layer, 'blocks': second_block, 'castto': 'Q4_1'}]   }
                    jobs.append((f"{first_layer}-{first_block} and {second_layer}-{second_block},,Q4_1", config, [first_layer, second_layer]))

    return jobs'''
    
def main():
    setup()
    the_data      = create_dataset()
    layer_stack   = load_layer_stack()
    jobs          = get_jobs_list()

    print(f"{len(jobs)} jobs")

    outfile = os.path.join(args.save_dir, args.results_file)
    if not os.path.exists(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open( outfile, 'a+' ) as output_filehandle:
        for job in jobs:
            result = job.execute(layer_stack, the_data)
            print(f"{result}")
            print(f"{result.label},{result.loss:>10.5},{result.time:>10.5}", file=output_filehandle, flush=True)

if __name__=='__main__': 
    main()
    shared.save_stats(filepath(args.save_dir,args.stats_file))
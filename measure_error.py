import add_paths
from modules.arguments import args, filepath
from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset
from modules.utils import Batcher, shared, is_double, load_config
from modules.casting import cast_layer_stack
import torch
from tqdm import tqdm, trange
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import os

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

def compute_loss(model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], autocast=False):
    img, txt, vec, pe, x_out = inputs['img'].cuda(), inputs['txt'].cuda(), inputs['vec'].cuda(), inputs['pe'].cuda(), inputs['x_out'].cuda()
    x = None
    loss_fn = torch.nn.MSELoss()

    with torch.autocast("cuda", enabled=autocast):
        for layer in model: 
            if isinstance(layer, DoubleStreamBlock): 
                img, txt = layer( img, txt, vec, pe ) 
            else:
                if x is None: x = torch.cat((txt, img), dim=1)
                x = layer( x, vec, pe )

    loss = float(loss_fn(x, x_out))
    print(f"Loss: {loss}")
    return(loss)

def setup():
    HFFS_Cache.set_cache_directory(args.cache_dir)
    shared.set_shared_filepaths(args=args)

    HFFS_Cache.set_cache_directory(args.cache_dir)
    Batcher.set_mode(all_in_one=True)
    MergedBatchDataset.set_dataset_source(dir=args.hs_dir)
    
def create_dataset():
    return MergedBatchDataset()

def load_model():
    print("Load model...")
    model = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    model.requires_grad_(False)
    return model

def modify_model(model, cast_config):
    cast_layer_stack(model, cast_config=cast_config, 
                        stack_starts_at_layer=0, default_cast=args.default_cast, 
                        verbose=args.verbose, autocast=args.autocast)
    
def copy_layer(model:torch.nn.Sequential, n):
    the_layer = new_layer(n)
    the_layer.load_state_dict( model[n].state_dict() )
    return the_layer

def restore_layer(model:torch.nn.Sequential, layer:torch.nn.Module, n):
    the_layer = new_layer(n)
    the_layer.load_state_dict( layer.state_dict() )
    model[n] = the_layer

def evaluate(model, dataset):
    model.cuda()
    with torch.no_grad():
        return [ compute_loss(model, entry) for entry in tqdm(dataset) ]
    
def main():
    setup()
    ds = create_dataset()
    model = load_model()

    for layer in range(19):
        saved_layer = copy_layer(model, layer)
        for blocks in ['txt', 'img']:
            for cast in ['Q8_0', 'Q5_1', 'Q4_1']:

                restore_layer(model, saved_layer, layer)
                modify_model(model, { 'casts': [{'layers': layer, 'blocks': blocks, 'castto': cast}] })
                mses = evaluate(model, ds)
                mean = sum(mses)/len(mses)
                with open("results.txt", 'a') as output:
                    print(f"{layer:>2},{blocks},{cast},{mean:>10.5}", file=output, flush=True)

    for layer in range(19, 57):
        saved_layer = copy_layer(model, layer)
        for blocks in ['']:
            for cast in ['Q8_0', 'Q5_1', 'Q4_1']:

                restore_layer(model, saved_layer, layer)
                modify_model(model, { 'casts': [{'layers': layer, 'blocks': blocks, 'castto': cast}] })
                mses = evaluate(model, ds)
                mean = sum(mses)/len(mses)
                with open("results.txt", 'a') as output:
                    print(f"{layer:>2},{blocks},{cast},{mean:>10.5}", file=output, flush=True)




if __name__=='__main__': 
    main()
    shared.save_stats(filepath(args.save_dir,args.stats_file))
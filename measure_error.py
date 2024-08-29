import add_paths
from modules.arguments import args, filepath
from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset
from modules.utils import Batcher, shared, is_double, load_config
from modules.casting import cast_layer_stack, QuantizedTensor
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
        for i, layer in enumerate(model): 
            if isinstance(layer, DoubleStreamBlock): 
                if isinstance(layer.txt_mlp[0].weight, QuantizedTensor):
                    print(f"Layer {i} has {layer.txt_mlp[0].weight.gtype.name}")
                img, txt = layer( img, txt, vec, pe ) 
            else:
                if x is None: x = torch.cat((txt, img), dim=1)
                x = layer( x, vec, pe )

    loss = float(loss_fn(x, x_out))
    #print(f"Loss: {loss}")
    return(loss)

def setup():
    HFFS_Cache.set_cache_directory(args.cache_dir)
    shared.set_shared_filepaths(args=args)

    HFFS_Cache.set_cache_directory(args.cache_dir)
    Batcher.set_mode(all_in_one=True)
    MergedBatchDataset.set_dataset_source(dir=args.hs_dir)
    
def create_dataset():
    return MergedBatchDataset(split='eval', eval_frac=0.1)

def load_model():
    print("Load model...")
    model = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    model.requires_grad_(False)
    return model

def modify_model(model, cast_config):
    cast_layer_stack(model, cast_config=cast_config, 
                        stack_starts_at_layer=0, default_cast=args.default_cast, 
                        verbose=args.verbose, autocast=args.autocast)
    
def clone_layer_sd(model:torch.nn.Sequential, n):
    sd:dict[str, torch.Tensor] = model[n].state_dict()
    return { k:sd[k].clone() for k in sd }

def restore_layer(model:torch.nn.Sequential, sd, n):
    the_layer = new_layer(n)
    the_layer.load_state_dict( sd )
    model = torch.nn.Sequential( *[m if i!=n else the_layer for i, m in enumerate(model)] )
    return model

def evaluate(model, dataset):
    model.cuda()
    with torch.no_grad():
        return [ compute_loss(model, entry) for entry in tqdm(dataset) ]
    
def main():
    setup()
    ds = create_dataset()
    model = load_model()

    for blocks in ['txt', 'img']:
        for cast in ['Q8_0', 'Q5_1', 'Q4_1']:
            for layer in range(19):
                saved_layer_sd = clone_layer_sd(model, layer)
                modify_model(model, { 'casts': [{'layers': layer, 'blocks': blocks, 'castto': cast}] })
                mses = evaluate(model, ds)
                mean = sum(mses)/len(mses)
                with open("results.txt", 'a') as output:
                    print(f"{layer:>2},{blocks},{cast},{mean:>10.5}", file=output, flush=True)
                model = restore_layer(model, saved_layer_sd, layer)





if __name__=='__main__': 
    main()
    shared.save_stats(filepath(args.save_dir,args.stats_file))
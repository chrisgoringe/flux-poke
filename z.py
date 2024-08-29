from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset
from modules.utils import Batcher, shared, is_double
import torch
import sys, os
from tqdm import tqdm, trange
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','flux','src'))
from flux.modules.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union

def load_single_layer(layer_number:int, remove_from_sd=True) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    layer_sd = shared.layer_sd(layer_number)
    if remove_from_sd: shared.drop_layer(layer_number)
    if is_double(layer_number):
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, qkv_bias=True)
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4)
    layer.load_state_dict(layer_sd)
    return layer

def compute_loss(model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], autocast=False):
    img, txt, vec, pe, x_out = inputs['img'].cuda(), inputs['txt'].cuda(), inputs['vec'].cuda(), inputs['pe'].cuda(), inputs['x_out'].cuda()
    loss_fn = torch.nn.MSELoss()

    with torch.autocast("cuda", enabled=autocast):
        for layer in tqdm(model): 
            if isinstance(layer, DoubleStreamBlock): 
                img, txt = layer( img, txt, vec, pe ) 
            else:
                if x is None: x = torch.cat((txt, img), dim=1)
                x = layer( x, vec, pe )

    return(float(loss_fn(x, x_out)))

def setup():
    torch.set_default_dtype(torch.bfloat16)
    shared._sd = "../flux1-dev.safetensors"
    HFFS_Cache.set_cache_directory("cache")
    Batcher.set_mode(all_in_one=True)
    MergedBatchDataset.set_dataset_source(dir="ChrisGoringe/fi2")
    
def create_dataset():
    return MergedBatchDataset()

def load_model():
    model = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    model.requires_grad_(False)
    model.cuda()
    return model

def modify_model(model):
    pass

def evaluate(model, dataset):
    with torch.no_grad():
        return [ compute_loss(model, entry) for entry in tqdm(dataset) ]
    
def main():
    setup()
    ds = create_dataset()
    model = load_model()
    modify_model(model)
    mses = evaluate(model, ds)
    pass

if __name__=='__main__': main()
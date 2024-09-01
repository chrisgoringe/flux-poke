import torch
import time
from collections.abc import Callable
from modules.casting import cast_layer_stack
from modules.pruning import prune_layer_stack
from comfy.ldm.flux.layers import DoubleStreamBlock
from modules.generated_dataset import _Dataset
from tqdm import tqdm

class Result:
    def __init__(self, label):
        self.label:str          = label
        self.time:float         = None
        self.losses:list[float] = None
        self.loss:float         = None

    @property
    def to_string(self):
        return f"{self.label}: loss {self.loss:>10.4f}, took {self.time:>10.4f}s"

class Job:
    layer_generator:Callable[[int,],torch.nn.Module] = None
    args = None

    def __init__(self, label:str, config:dict, preserve_layers:list[int]=[], prerun:Callable=None, postrun:Callable=None, callbacks:list[Callable]=[]):
        self.config = config
        self.preserve_layers = preserve_layers
        self.prerun = prerun
        self.postrun = postrun
        self.result = Result(label)
        self.callbacks = callbacks

        self.default_cast = self.args.default_cast
        self.autocast:bool = self.args.autocast
        self.verbose:int = self.args.verbose

    def execute(self, layer_stack:torch.nn.Sequential, the_data) -> tuple[Result, torch.nn.Sequential]:
        if self.prerun: self.prerun()

        saved_layer_sds = [ self.clone_layer_sd(layer_stack, layer_index) for layer_index in self.preserve_layers ]
        self.modify_layer_stack( layer_stack, 
                                cast_config  = self.config if 'casts'  in self.config else None,
                                prune_config = self.config if 'prunes' in self.config else None,
                                patch_config = None )
        
        layer_stack.cuda()
        start_time = time.monotonic()
        losses = self.evaluate(layer_stack, the_data)
        self.result.time   = time.monotonic() - start_time
        self.result.losses = losses
        self.result.loss   = sum(losses) / len(losses)
        layer_stack.cpu()

        for i, layer_index in enumerate(self.preserve_layers):
            layer_stack = self.restore_layer(layer_stack, sd=saved_layer_sds[i], layer_number=layer_index)

        if self.postrun: self.postrun()
        return self.result, layer_stack
    
    def clone_layer_sd(self, layer_stack:torch.nn.Sequential, layer_number) -> dict[str,torch.Tensor]:
        sd:dict[str, torch.Tensor] = layer_stack[layer_number].state_dict()
        return { k:sd[k].clone() for k in sd }

    def restore_layer(self, layer_stack:torch.nn.Sequential, sd, layer_number) -> torch.nn.Sequential:
        the_layer = self.layer_generator(layer_number)
        the_layer.load_state_dict( sd )
        layer_stack = torch.nn.Sequential( *[m if i!=layer_number else the_layer for i, m in enumerate(layer_stack)] )
        return layer_stack
    
    def modify_layer_stack(self, layer_stack:torch.nn.Sequential, cast_config, prune_config, patch_config):
        if cast_config:
            print(cast_config)
            cast_layer_stack(layer_stack, cast_config=cast_config, 
                                stack_starts_at_layer=0, default_cast=self.default_cast, 
                                verbose=self.verbose, autocast=self.autocast)
        if prune_config:
            prune_layer_stack(layer_stack, prune_config=prune_config, model_first_layer=0, verbose=self.verbose)
        if patch_config:
            raise NotImplementedError()
        
    def evaluate(self, layer_stack, dataset:_Dataset):
        with torch.no_grad():
            losses = []
            for entry in tqdm(dataset):
                loss = compute_loss(layer_stack, entry, self.autocast)
                for callback in self.callbacks: callback(loss, dataset.last_was())
                losses.append(loss)
            return losses

def compute_loss(model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], autocast=False, loss_fn=None) -> float:
    img, txt, vec, pe, x_out = inputs['img'].cuda(), inputs['txt'].cuda(), inputs['vec'].cuda(), inputs['pe'].cuda(), inputs['x_out'].cuda()
    x = None
    loss_fn = loss_fn or torch.nn.MSELoss()

    for i, layer in enumerate(model): 
        if isinstance(layer, DoubleStreamBlock): 
            with torch.autocast("cuda", enabled=autocast): img, txt = layer( img, txt, vec, pe ) 
        else:
            if x is None: x = torch.cat((txt, img), dim=1)
            with torch.autocast("cuda", enabled=autocast): x = layer( x, vec, pe )

    return float(loss_fn(x, x_out))
import transformers
import torch
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from .utils import int_list_from_string, shared
from .arguments import args

class TheTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()
        self.can_return_loss = True

    def check_inf(self,*args):
        for a in args:
            if torch.isinf(a).any():
                print("INF")

    def compute_loss(self, model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], return_outputs=False):
        vec = inputs.get('vec', None)
        pe  = inputs.get('pe',  None)
        x   = inputs.get('x',   None)
        img = inputs.get('img', None)
        txt = inputs.get('txt', None)

        with torch.autocast("cuda", enabled=args.autocast):
            for layer in model: 
                if isinstance(layer, DoubleStreamBlock): 
                    img, txt = layer( img, txt, vec, pe ) 
                    self.check_inf(img, txt)
                else:
                    if x is None: x = torch.cat((txt, img), dim=1)
                    x = layer( x, vec, pe )
                    self.check_inf(x)

            if 'img_out' in inputs:
                loss = self.loss_fn(torch.cat((txt, img), dim=1), torch.cat((inputs['txt_out'], inputs['img_out']), dim=1))
                return (loss, (img, txt)) if return_outputs else loss
            else:
                if x is None: x = torch.cat((txt, img), dim=1)
                loss = self.loss_fn(x, inputs['x_out'])
                return (loss, x) if return_outputs else loss

def prep_layer_for_train(layer, block_constraint, callback):
    def recurse(parent_name:str, child_module:torch.nn.Module, child_name:str):
        child_fullname = ".".join((parent_name,child_name)) if parent_name else child_name
        if isinstance(child_module, torch.nn.Linear):
            if block_constraint is None or block_constraint in child_fullname:
                child_module.requires_grad_(True)
                if callback: callback(child_fullname)
        else:
            for grandchild_name, grandchild_module in child_module.named_children():
                recurse(child_fullname, grandchild_module, grandchild_name)

    for child_name, child_module in layer.named_children():
        recurse("", child_module, child_name)

training_count = 0
def prep_for_train(model, train_config, layer_index, verbose):
    global training_count
    training_count = 0
    for mod in train_config.get('trains',None) or []:
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint = None
        if block_constraint != 'none':
            for global_layer_index in int_list_from_string(mod.get('layers',None)):
                model_layer_index = global_layer_index - layer_index
                if model_layer_index>=0 and model_layer_index<len(model):
                    layer = model[model_layer_index]
                    def record(linear_name): 
                        global training_count
                        training_count += 1
                        if verbose: print(f"{global_layer_index}.{linear_name} set for training")
                        shared.layer_stats[global_layer_index][linear_name] = "Set for training"
                    prep_layer_for_train(layer=layer, block_constraint=block_constraint, callback=record)
    return training_count


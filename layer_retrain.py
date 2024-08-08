import sys, os, json, yaml
sys.path.insert(0,os.getcwd())

from modules.modifiers import slice_double_block, get_mask
from comfy.ldm.flux.layers import DoubleStreamBlock
import torch
from safetensors.torch import load_file, save_file
import transformers
from modules.arguments import args, filepath

from modules.utils import log, shared

class TheTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()
        self.can_return_loss = True

    def compute_loss(self, model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], return_outputs=False):
        img, txt, vec, pe, img_out, txt_out = inputs['img'], inputs['txt'], inputs['vec'], inputs['pe'], inputs['img_out'],inputs['txt_out']
        for layer in model: img, txt = layer( img, txt, vec, pe ) 
        loss = self.loss_fn(torch.cat((img,txt),dim=1), torch.cat((img_out, txt_out), dim=1))
        return (loss, (img, txt)) if return_outputs else loss
    
class TheCallback(transformers.TrainerCallback):
    log:list[dict] = None
    def on_evaluate(self, _, state: transformers.TrainerState, *args, **kwargs):  TheCallback.log = state.log_history.copy()
    def on_train_end(self, _, state: transformers.TrainerState, *args, **kwargs): TheCallback.log = state.log_history.copy()

    @classmethod
    def eval_losses(cls): return [l['eval_loss'] for l in cls.log if 'eval_loss' in l]
    @classmethod
    def losses(cls): return [l['loss'] for l in cls.log if 'loss' in l]
    
class TheDataset:
    def __init__(self, first_layer:int, split:str, thickness:int=1, train_frac=0.8):
        dir = filepath(args.hs_dir)
        self.sources = [ os.path.join(dir,x) for x in os.listdir(dir) if x.endswith(".safetensors") ]
        split_at = int(train_frac*len(self.sources))
        if   split=='train': self.sources = self.sources[:split_at]
        elif split=='eval':  self.sources = self.sources[split_at:]
        self.layer = first_layer
        self.thickness = thickness
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):
        all_data = load_file(self.sources[i])
        return {
            "img"     : all_data[f"{self.layer}-img"].squeeze(0),
            "txt"     : all_data[f"{self.layer}-txt"].squeeze(0),
            "vec"     : all_data[f"{self.layer}-vec"].squeeze(0),
            "pe"      : all_data[f"{self.layer}-pe"].squeeze(0),
            "img_out" : all_data[f"{self.layer+self.thickness}-img"].squeeze(0),
            "txt_out" : all_data[f"{self.layer+self.thickness}-txt"].squeeze(0),
        }

def load_pruned_layer(layer_number:int) -> DoubleStreamBlock:
    log(f"Loading and pruning layer {layer_number}")
    layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    prefix = f"double_blocks.{layer_number}."
    layer_sd = { k[len(prefix):]:shared.sd[k] for k in shared.sd if k.startswith(prefix) }
    layer.load_state_dict(layer_sd)

    img_mask, img_threshold = get_mask(shared.internals[f"double-img-{layer_number}"], args.img_threshold, args.img_count, return_threshold=True) 
    txt_mask, txt_threshold = get_mask(shared.internals[f"double-txt-{layer_number}"], args.txt_threshold, args.txt_count, return_threshold=True)

    slice_double_block(layer, img_mask=img_mask, txt_mask=txt_mask)

    layer.requires_grad_(False)
    layer.img_mlp.requires_grad_(True)
    layer.txt_mlp.requires_grad_(True)

    shared.layer_stats[layer_number]['img_threshold'] = int(img_threshold)
    shared.layer_stats[layer_number]['txt_threshold'] = int(txt_threshold)
    shared.layer_stats[layer_number]['img_lines']     = int(layer.img_mlp[0].out_features)
    shared.layer_stats[layer_number]['txt_lines']     = int(layer.txt_mlp[0].out_features)

    return layer

def train_layer(layer_index:int, thickness:int):
    log(f"Model is layers {layer_index} to {layer_index+thickness-1}" if thickness > 1 else f"Model is layer {layer_index}")
    assert layer_index+thickness-1 <= shared.max_layer, f"max_layer ({shared.max_layer}) exceeded"

    model = torch.nn.Sequential(
        *[load_pruned_layer(layer_number=x) for x in range(layer_index, layer_index+thickness)]
    )
    log(str(shared.layer_stats[layer_index]))

    t = TheTrainer(
        model         = model,
        args          = args.training_args,
        train_dataset = TheDataset(first_layer=layer_index, split="train", thickness=1),
        eval_dataset  = TheDataset(first_layer=layer_index, split="eval",  thickness=1),
        data_collator = transformers.DefaultDataCollator(),
        callbacks     = [TheCallback,],
    )

    if args.just_evaluate:
        t.evaluate()
        shared.layer_stats[layer_index]['initial_loss'] = TheCallback.eval_losses()[0]
    else:
        t.train()
        shared.layer_stats[layer_index]['initial_loss'] = TheCallback.eval_losses()[0]
        shared.layer_stats[layer_index]['final_loss']   = TheCallback.eval_losses()[-1]
        shared.layer_stats[layer_index]['train_loss']   = TheCallback.losses()[-1]
        for i,layer in enumerate(model):
            savefile = filepath(args.save_dir,f"{layer_index+i}.safetensors")
            save_file(layer.to(args.save_dtype).state_dict(), savefile)
            log(f"Saved in {savefile}")

    log(str(shared.layer_stats[layer_index]))

        
if __name__=="__main__": 
    start_layer_list = [int(x.strip()) for x in args.first_layer.split(',')]
    for l in start_layer_list: train_layer(l, args.thickness)
    with open(filepath(args.save_dir,args.stats_yaml), 'w') as f: print(shared.layer_stats_yaml, file=f)
    log(f"Saved stats in {filepath(args.save_dir,args.stats_yaml)}")
    for l in shared.layer_stats: log( str(l) )    

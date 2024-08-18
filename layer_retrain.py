import sys, os
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.join(os.getcwd(),'..','..'))

from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from safetensors.torch import save_file
import transformers
from modules.arguments import args, filepath
from modules.generated_dataset import TheDataset
from modules.utils import log, shared, int_list_from_string, load_config, is_double
from modules.hffs import HFFS_Cache
from modules.trainer import TheTrainer, prep_for_train
from modules.casting import cast_model
from modules.pruning import prune_model
from typing import Union

class TheCallback(transformers.TrainerCallback):
    log:list[dict] = None
    def on_evaluate(self, _, state: transformers.TrainerState, *args, **kwargs):  TheCallback.log = state.log_history.copy()
    def on_train_end(self, _, state: transformers.TrainerState, *args, **kwargs): TheCallback.log = state.log_history.copy()

    @classmethod
    def eval_losses(cls): return [l['eval_loss'] for l in cls.log if 'eval_loss' in l]
    @classmethod
    def losses(cls): return [l['loss'] for l in cls.log if 'loss' in l]

def load_unpruned_layer(layer_number:int) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    log(f"Loading layer {layer_number}")
    if is_double(layer_number):
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
        prefix = f"double_blocks.{layer_number}."
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)
        prefix = f"single_blocks.{layer_number - shared.last_double_layer - 1}."

    layer_sd = { k[len(prefix):]:shared.sd[k] for k in shared.sd if k.startswith(prefix) }
    layer.load_state_dict(layer_sd)
    return layer

def set_trainable(layer, layer_index):
    if is_double(layer_index):
        layer.img_mlp.requires_grad_(True)
        layer.txt_mlp.requires_grad_(True)
    else:
        layer.linear1.requires_grad_(True)
        layer.linear2.requires_grad_(True)        

def train_layer():
    log(f"Model is layers {args.first_layer} to {args.first_layer+args.thickness-1}" if args.thickness > 1 else f"Model is layer {args.first_layer}")
    assert args.first_layer+args.thickness-1 <= shared.max_layer, f"max_layer ({shared.max_layer}) exceeded"

    model = torch.nn.Sequential( *[load_unpruned_layer(layer_number=x) for x in range(args.first_layer, args.first_layer+args.thickness)] )
    model.requires_grad_(False)

    if args.prune_map:
        prune_config = load_config(filepath(args.prune_map))
        prune_model(model, prune_config=prune_config, layer_index=args.first_layer, )

    if args.cast_map:
        cast_config = load_config(filepath(args.cast_map))
        cast_model(model, cast_config=cast_config, layer_index=args.first_layer, default_cast=args.default_cast)

    if args.train_map:
        train_config = load_config(filepath(args.train_map))
        prep_for_train(model, train_config=train_config, layer_index=args.first_layer)
    else:
        if not args.just_evaluate:
            print("No train_map specified. Switching to just_evaluate")
            args.just_evaluate = True
        
        
    t = TheTrainer(
        model         = model,
        args          = args.training_args,
        train_dataset = TheDataset(dir=args.hs_dir, first_layer=args.first_layer, split="train", train_frac=args.train_frac, thickness=args.thickness),
        eval_dataset  = TheDataset(dir=args.hs_dir, first_layer=args.first_layer, split="eval",  train_frac=args.train_frac, thickness=args.thickness),
        data_collator = transformers.DefaultDataCollator(),
        callbacks     = [TheCallback,],
    )

    if args.just_evaluate:
        t.evaluate()
        shared.layer_stats[args.first_layer]['initial_loss'] = TheCallback.eval_losses()[0]
    else:
        t.train()
        shared.layer_stats[args.first_layer]['initial_loss'] = TheCallback.eval_losses()[0]
        shared.layer_stats[args.first_layer]['final_loss']   = TheCallback.eval_losses()[-1]
        shared.layer_stats[args.first_layer]['train_loss']   = TheCallback.losses()[-1]
        for i,layer in enumerate(model):
            savefile = filepath(args.save_dir,f"{args.first_layer+i}.safetensors")
            save_file(layer.to(args.save_dtype).state_dict(), savefile)
            log(f"Saved in {savefile}")

    log(str(shared.layer_stats[args.first_layer]))

        
if __name__=="__main__": 
    HFFS_Cache.set_cache_directory(args.cache_dir)
    if args.clear_cache_before: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()

    for l in int_list_from_string(args.first_layers or args.first_layer): 
        log("Loading vanilla model")
        shared.load(args=args)
        args.first_layer = l
        train_layer()

    shared.save_stats(filepath(args.save_dir,args.stats_file))

    if args.clear_cache_after: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()
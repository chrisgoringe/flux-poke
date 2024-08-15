import sys, os
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.join(os.getcwd(),'..','..'))

from modules.modifiers import slice_double_block, get_mask, slice_single_block
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
import torch
from safetensors.torch import save_file
import transformers
from modules.arguments import args, filepath
from modules.generated_dataset import TheDataset
from modules.utils import log, shared, range_from_string
from modules.hffs import HFFS_Cache
from modules.trainer import TheTrainer
from modules.casting import cast_model
from typing import Union

class TheCallback(transformers.TrainerCallback):
    log:list[dict] = None
    def on_evaluate(self, _, state: transformers.TrainerState, *args, **kwargs):  TheCallback.log = state.log_history.copy()
    def on_train_end(self, _, state: transformers.TrainerState, *args, **kwargs): TheCallback.log = state.log_history.copy()

    @classmethod
    def eval_losses(cls): return [l['eval_loss'] for l in cls.log if 'eval_loss' in l]
    @classmethod
    def losses(cls): return [l['loss'] for l in cls.log if 'loss' in l]

def load_pruned_layer(layer_number:int) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    log(f"Loading and pruning layer {layer_number}")
    is_double = (layer_number <= shared.last_double_layer)
    if is_double:
        layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
        prefix = f"double_blocks.{layer_number}."
    else:
        layer = SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)
        prefix = f"single_blocks.{layer_number - shared.last_double_layer - 1}."

    layer_sd = { k[len(prefix):]:shared.sd[k] for k in shared.sd if k.startswith(prefix) }
    layer.load_state_dict(layer_sd)
    layer.requires_grad_(False)

    if is_double:
        if not (args.img_no and args.txt_no):
            img_mask, img_threshold = get_mask(shared.internals["{:0>2}-img".format(layer_number)], args.img_threshold, args.img_count, args.img_no, return_threshold=True) 
            txt_mask, txt_threshold = get_mask(shared.internals["{:0>2}-txt".format(layer_number)], args.txt_threshold, args.txt_count, args.txt_no, return_threshold=True)
            slice_double_block(layer, img_mask=img_mask, txt_mask=txt_mask)
            if not args.img_no:
                shared.layer_stats[layer_number]['img_threshold'] = int(img_threshold)
                shared.layer_stats[layer_number]['img_lines']     = int(layer.img_mlp[0].out_features)
            if not args.txt_no:
                shared.layer_stats[layer_number]['txt_threshold'] = int(txt_threshold)
                shared.layer_stats[layer_number]['txt_lines']     = int(layer.txt_mlp[0].out_features)

        if not args.just_evaluate:
            layer.img_mlp.requires_grad_(True)
            layer.txt_mlp.requires_grad_(True)

    else:
        if not args.x_no:
            mask, x_threshold = get_mask(shared.internals["{:0>2}-x".format(layer_number)], args.x_threshold, args.x_count, args.x_no, return_threshold=True) 
            slice_single_block(layer, mask=mask)

        if not args.just_evaluate:
            layer.linear1.requires_grad_(True)
            layer.linear2.requires_grad_(True)
            shared.layer_stats[layer_number]['x_threshold'] = int(x_threshold)
            shared.layer_stats[layer_number]['x_lines']     = int(layer.linear1.out_features)

    return layer

def train_layer(layer_index:int, thickness:int):
    log(f"Model is layers {layer_index} to {layer_index+thickness-1}" if thickness > 1 else f"Model is layer {layer_index}")
    assert layer_index+thickness-1 <= shared.max_layer, f"max_layer ({shared.max_layer}) exceeded"

    model = torch.nn.Sequential(
        *[load_pruned_layer(layer_number=x) for x in range(layer_index, layer_index+thickness)]
    )
    log(str(shared.layer_stats[layer_index]))

    if args.cast_map:
        log("Casting")
        cast_model(model, cast_map_filepath=filepath(args.cast_map), layer_index=layer_index, thickness=thickness)
        
    t = TheTrainer(
        model         = model,
        args          = args.training_args,
        train_dataset = TheDataset(dir=args.hs_dir, first_layer=layer_index, split="train", train_frac=args.train_frac, thickness=1),
        eval_dataset  = TheDataset(dir=args.hs_dir, first_layer=layer_index, split="eval",  train_frac=args.train_frac, thickness=1),
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
    HFFS_Cache.set_cache_directory(args.cache_dir)
    if args.clear_cache_before: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()

    for l in range_from_string(args.first_layer): 
        log("Loading vanilla model")
        shared.load(args=args)
        train_layer(l, args.thickness)

    shared.save_stats(filepath(args.save_dir,args.stats_yaml))

    if args.clear_cache_after: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()
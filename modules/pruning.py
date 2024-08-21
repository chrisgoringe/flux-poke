from .arguments import args
from .utils import shared, is_double, int_list_from_string, prefix
from modules.modifiers import slice_double_block, get_mask, slice_single_block
import os
from safetensors.torch import load_file

def prune_layer(layer, global_layer_number:int, count, constraint, callbacks=[]):   
    if not count: return
    
    if is_double(global_layer_number):
        do_img = constraint is None or 'img' in constraint
        do_txt = constraint is None or 'txt' in constraint
        img_mask, img_threshold = get_mask(shared.internals["{:0>2}-img".format(global_layer_number)], remove_count=count if do_img else None) 
        txt_mask, txt_threshold = get_mask(shared.internals["{:0>2}-txt".format(global_layer_number)], remove_count=count if do_txt else None)
        slice_double_block(layer, img_mask=img_mask, txt_mask=txt_mask)
        if do_img: 
            for callback in callbacks: callback('double_blocks','img_mlp', count, img_threshold)
        if do_txt:
            for callback in callbacks: callback('double_blocks','txt_mlp', count, txt_threshold)
    else:
        mask, x_threshold = get_mask(shared.internals["{:0>2}-x".format(global_layer_number)], remove_count=count) 
        slice_single_block(layer, mask=mask)
        for callback in callbacks: 
            callback('single_blocks','linear1', count, x_threshold)
            callback('single_blocks','linear2', count, x_threshold)

def prune_model(model, prune_config, model_first_layer, verbose, callbacks=[]):
    for mod in prune_config.get('prunes',None) or []:
        remove = mod.get('remove',0)
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint = None
        if remove and block_constraint != 'none':
            for global_layer_number in int_list_from_string(mod.get('layers',None)):
                model_layer_index = global_layer_number - model_first_layer
                if model_layer_index>=0 and model_layer_index<len(model):
                    layer = model[model_layer_index]
                    def record(parent,block, number, threshold): 
                        if verbose: print(f"{parent}.{global_layer_number}.{block} pruned by {number} (threshold {threshold})")
                        shared.layer_stats[global_layer_number][block] = f"Pruned by {number} (threshold {threshold})"
                    prune_layer(layer, global_layer_number=global_layer_number, count=remove, constraint=block_constraint, callbacks=callbacks+[record,])

def apply_patches(sd, from_directory, callbacks=[]):
    def patch_list():
        for x in range(57): 
            path = os.path.join(from_directory,"{:0>2}.safetensors".format(x))
            if os.path.exists(path): yield x, path

    for patch_layer_index, patch_filepath in patch_list():
        patch = load_file(patch_filepath)
        for k in patch:
            assert (key:=f"{prefix(patch_layer_index)}{k}") in sd
            sd[key] = patch[k]
            for callback in callbacks: callback(key)
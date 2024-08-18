from .arguments import args
from .utils import shared, is_double, int_list_from_string
from modules.modifiers import slice_double_block, get_mask, slice_single_block

def prune_layer(layer, layer_number:int, count, constraint, callback):   

    if is_double(layer_number):
        skip_img = constraint and 'txt' in constraint
        skip_txt = constraint and 'img' in constraint
        img_mask, img_threshold = get_mask(shared.internals["{:0>2}-img".format(layer_number)], remove_count=count, remove_nothing=skip_img, return_threshold=True) 
        txt_mask, txt_threshold = get_mask(shared.internals["{:0>2}-txt".format(layer_number)], remove_count=count, remove_nothing=skip_txt, return_threshold=True)
        slice_double_block(layer, img_mask=img_mask, txt_mask=txt_mask)
        callback('img', img_mask, img_threshold)
        callback('txt', txt_mask, txt_threshold)
    else:
        mask, x_threshold = get_mask(shared.internals["{:0>2}-x".format(layer_number)], remove_count=count, return_threshold=True) 
        slice_single_block(layer, mask=mask)
        callback('x', mask, x_threshold)

def prune_model(model, prune_config, layer_index):
    for mod in prune_config.get('prunes',None) or []:
        remove = mod.get('remove',0)
        if (block_constraint:=mod.get('blocks', 'all')) == 'all': block_constraint = None
        if remove and block_constraint != 'none':
            for global_layer_index in int_list_from_string(mod.get('prunes',None)):
                model_layer_index = global_layer_index - layer_index
                if model_layer_index>=0 and model_layer_index<len(model):
                    layer = model[model_layer_index]
                    def record(block, number, threshold): shared.layer_stats[global_layer_index][block] = f"Pruned by {number} (threshold {threshold})"
                    prune_layer(layer, model_layer_index, remove, block_constraint, record)
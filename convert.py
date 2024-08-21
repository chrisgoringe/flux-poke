import add_paths
from modules.arguments import args
from safetensors.torch import load_file, save_file
from modules.utils import filepath, load_config, shared, prefix
from modules.casting import cast_layer_stack
from modules.layer import load_single_layer
from modules.pruning import apply_patches, prune_model
import torch

def convert():
    assert args.saved_model

    # load state directory
    sd = load_file(args.model)

    # replace with loaded patches
    if args.load_patches:
        patched = []
        def record(key): patched.append(key)
        for dir in args.load_patches: apply_patches(sd, filepath(dir), [record,])
        assert (len(patched)==len(set(patched)) or args.allow_overpatching), "Loaded patches overlapped. To allow this use --allow_overpatching"

    # load layers
    shared._sd = sd
    all_layers = torch.nn.Sequential( *[load_single_layer(layer_number=x, remove_from_sd=True) for x in range(0, shared.last_layer+1)] )
    # Only keep what isn't in one of the layers:
    # sd = { k:sd[k] for k in sd if '_blocks.' not in k }

    # prune layers - hopefully not ones we've patched....
    if args.prune_map:
        pruned = []
        def record(parent,block, number, threshold): pruned.append(f"{parent}{block}")
        prune_config = load_config(filepath(args.prune_config))
        prune_model(all_layers, prune_config, model_first_layer=0, verbose=args.verbose, callbacks=[record,])

    # cast layers
    if args.cast_map:
        cast_config = load_config(filepath(args.cast_map))
        all_casts = {}
        def track_casts(name, type): all_casts[name] = str(type)
        cast_layer_stack(layer_stack=all_layers, cast_config=cast_config, stack_starts_at_layer=0,
                         default_cast=args.default_cast, verbose=True, callbacks=[track_casts,])

    # create final sd
    # Then the layers
    for layer_index, layer in enumerate(all_layers):
        for k in (layer_sd:=layer.state_dict()):
            sd[f"{prefix(layer_index)}{k}"] = layer_sd[k]
        all_layers[layer_index] = None  # free up memory

    save_file(sd, args.saved_model)

if __name__=='__main__': 
    convert()
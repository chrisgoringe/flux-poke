from modules.arguments import args
from safetensors.torch import load_file, save_file
from modules.utils import filepath, load_config, shared, prefix
from modules.casting import cast_layer_stack
from modules.layer import load_single_layer
from modules.pruning import apply_patches
import json, torch

def convert():
    assert args.saved_model

    # load state directory
    sd = load_file(args.model)

    # patch it
    if args.load_patches:
        patched = []
        def record(key): patched.append(key)
        for dir in args.load_patches:
            apply_patches(sd, filepath(dir), record)
        assert len(patched)==len(set(patched))

    # load layers
    shared._sd = sd
    all_layers = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in range(args.first_layer, args.first_layer+args.thickness)] )
        
    # cast layers
    if args.cast_map:
        cast_config = load_config(filepath(args.cast_map))
        casting_metadata = json.dumps(cast_config['casts'])
        if args.default_cast and args.default_cast != 'no':
            casting_metadata = casting_metadata.replace('default',args.default_cast)

        all_casts = {}
        def track_casts(name, type): all_casts[name] = str(type)
        cast_layer_stack(layer_stack=all_layers, cast_config=cast_config, stack_starts_at_layer=0,
                         default_cast=args.default_cast, verbose=True, callbacks=[track_casts,])

    # create final sd
    sd = { k:sd[k] for k in sd if '_blocks.' not in k }
    for layer_index, layer in enumerate(all_layers):
        for k in (layer_sd:=layer.state_dict()):
            sd[f"{prefix(layer_index)}{k}"] = layer_sd[k]

    save_file(sd, args.saved_model)

if __name__=='__main__': 
    convert()
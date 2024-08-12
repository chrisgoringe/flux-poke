from argparse import ArgumentParser
import torch, transformers, os
from modules.utils import filepath, shared

args = None

def process_arguments():
    a = ArgumentParser(fromfile_prefix_chars='@')
    a.add_argument('--first_layer', required=True, help="The first layer to be trained. Comma separated list to loop over layers.")
    a.add_argument('--thickness', type=int, default=1, help="The thickness to be trained (default 1)")
    a.add_argument('--model', type=str, required=True, help="flux dev model (absolute path)")
    a.add_argument('--save_dir', default="retrained_layers", help="directory, relative to cwd, to store results in")
    a.add_argument('--stats_yaml', default="layer_stats.yaml", help="filename (relative to save_dir) for stats to be saved in")
    a.add_argument('--cache_dir', default=None, help="If using HFFS, where to cache files")
    a.add_argument('--clear_cache_before', action="store_true", help="Clear the cache at the start of the run" )
    a.add_argument('--clear_cache_after', action="store_true", help="Clear the cache at the start of the run" )

    img = a.add_mutually_exclusive_group(required=True)
    img.add_argument('--img_threshold', type=int, help="Threshold below which img lines are dropped")
    img.add_argument('--img_count', type=int, help="Number of img lines to discard (of 12288)")
    img.add_argument('--img_no', action="store_true", help="Don't discard any img lines")

    txt = a.add_mutually_exclusive_group(required=True)
    txt.add_argument('--txt_threshold', type=int, help="Threshold below which txt lines are dropped")
    txt.add_argument('--txt_count', type=int, help="Number of txt lines to discard (of 12288)")
    txt.add_argument('--txt_no', action="store_true", help="Don't discard any txt lines")

    a.add_argument('--hs_dir', default="hidden_states", help="directory, relative to cwd, data is found in")
    a.add_argument('--save_dtype', default="bfloat16", choices=["bfloat16", "float8_e4m3fn", "float8_e5m2", "float16", "float"], help="tensor dtype to save" )
    a.add_argument('--internals', default="internals.safetensors", help="internals file, relative to cwd")

    a.add_argument('--just_evaluate', action='store_true', help='no training, just calculate the loss caused by the pruning')

    args, extra_args = a.parse_known_args([f"@{filepath('arguments.txt')}",])

    for k in ['img_threshold', 'img_count', 'txt_threshold', 'txt_count']: 
        if not hasattr(args,k): setattr(args, k, None)

    training_config = {
        "save_strategy"    : "no",
        "eval_strategy"    : "steps",
        "logging_strategy" : "steps",
        "output_dir"       : "output",
        "eval_on_start"    : True,
        "lr_scheduler_type": "cosine",
        "max_steps"        : 1000,
        "logging_steps"    :   50,
        "eval_steps"       :  100,
        "per_device_train_batch_size" : 8,
        "per_device_eval_batch_size"  : 8,
        "learning_rate"               : 5e-5,
        "remove_unused_columns" : False,
        "label_names"           : [],
    }

    for ea in extra_args:
        key, value = ea[2:].split('=')
        dictionary = training_config
        if ':' in key: 
            sub, key = key.split(':')
            if not sub in dictionary: dictionary[sub] = {}
            dictionary = training_config[sub]
        if   (value.lower()=='true'):  dictionary[key]=True
        elif (value.lower()=='false'): dictionary[key]=False
        else:
            try: 
                dictionary[key] = int(value)
            except ValueError:
                try:
                    dictionary[key] = float(value)
                except ValueError:
                    dictionary[key] = value

    setattr(args, 'training_args', transformers.TrainingArguments(**training_config))

    args.save_dtype = getattr(torch, args.save_dtype)
    if not os.path.exists(filepath(args.save_dir)): os.makedirs(filepath(args.save_dir), exist_ok=True)

    return args

if args is None: 
    args = process_arguments()
    shared.load(args)
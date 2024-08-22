from argparse import ArgumentParser
import torch, transformers, os
from modules.utils import filepath, shared


class CommentArgumentParser(ArgumentParser):
    '''
    An argument parser which:
    allows lines to be commented out with #,
    allows lines to have comments at the end starting #
    ignores blank lines, 
    strips whitespace and quotation marks around each side of any '='
    allows substitutions defined @label=value to then be used as @label
    allows conditional lines starting ?label which are ignored if @label is not defined or is blank
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines = []
        self.subs:dict[str,str] = {}
        self.strips = ' \"\'\t'

    def split_and_strip(self, s:str) -> list[str]:
        return list(a.strip(self.strips) for a in s.split('='))

    def convert_arg_line_to_args(self, arg_line:str) -> list[str]:
        arg_line = arg_line.split('#')[0]

        if arg_line.startswith('@'):
            search, replace = self.split_and_strip(arg_line)
            if replace: 
                for a in self.subs: replace = replace.replace(a, self.subs[a])        
                self.subs[search] = replace
            return []
        
        if arg_line.startswith('?'):
            condition, arg_line = arg_line.split(' ',1)
            if "@"+condition[1:] not in self.subs: return []

        for a in self.subs: arg_line = arg_line.replace(a, self.subs[a])
        if '@' in arg_line: print(f"{arg_line} contains undefined @")

        line = "=".join(self.split_and_strip(arg_line))
        if len(line): self.lines.append(line)
        return [line,] if len(line) else []
    
args = None

def process_arguments():
    a = CommentArgumentParser(fromfile_prefix_chars='@')
    first = a.add_mutually_exclusive_group(required=True)
    first.add_argument('--first_layer', help="The first layer in the sub-model.")
    first.add_argument('--first_layers', help="The range of first layers in the submodel. The whole process runs for each in the range.")

    a.add_argument('--thickness', type=int, default=1, help="The thickness to be trained (default 1)")
    a.add_argument('--model', type=str, required=True, help="flux dev model (absolute path)")
    a.add_argument('--load_patches', action="append", type=str, help="directory to load existing patches from (can have multiple)")
    a.add_argument('--allow_overpatching', action="store_true", help="allow one patch to overwrite a previous one (default is an assertion fail)")
    a.add_argument('--save_dir', default="output", help="Relative path of directory to store results in (includes patches if training)")
    a.add_argument('--saved_model', type=str, help="saved model when doing convert (absolute path)")

    a.add_argument('--verbose', action="store_true")
    a.add_argument('--run_asyncs', action="store_true", help="Experimental; try to cast asynchronously")
    a.add_argument('--autocast', action="store_true", help="Use autocast")
    
    a.add_argument('--cast_map', default=None, help="Relative path to yaml/json file describing how the layers should be cast")
    a.add_argument('--prune_map', default=None, help="Relative path to yaml/json file describing how the layers should be pruned")
    a.add_argument('--train_map', default=None, help="Relative path to yaml/json file describing how the layers should be trained")
    a.add_argument('--sensitive_hack', help="IMG, TXT or X - sensitivity hack")
    a.add_argument('--internals', default="internals.safetensors", help="Relative path of internals file")
    a.add_argument('--default_cast', default="bfloat16", help="Cast to use for 'default' in the cast_map")

    
    a.add_argument('--stats_file', default="stats.yaml", help="Path for stats to be saved in relative to save_dir")
    a.add_argument('--cache_dir', default=None, help="If using HFFS, where to cache files (absolute path)")
    a.add_argument('--clear_cache_before', action="store_true", help="Clear the cache at the start of the run" )
    a.add_argument('--clear_cache_after', action="store_true", help="Clear the cache at the start of the run" )

    a.add_argument('--hs_dir', default="hidden_states", help="Relative path of directory data is found in, or repo_id")
    fraction = a.add_mutually_exclusive_group()
    fraction.add_argument('--train_frac', type=float, help="fraction of dataset to train on")
    fraction.add_argument('--eval_frac', type=float, help="fraction of dataset to evalaute on")
    a.add_argument('--shuffle', action='store_true', help="shuffle the dataset")
    a.add_argument('--shuffle_seed', default=42)

    act = a.add_mutually_exclusive_group(required=True)
    act.add_argument('--evaluate', action='store_true', help='no training, just calculate the loss caused by the pruning')
    act.add_argument('--train', action='store_true')
    act.add_argument('--convert', action='store_true')

    args, extra_args = a.parse_known_args([f"@{filepath('arguments.txt')}",])

    if args.train_frac is None:
        args.train_frac = 1 - args.eval_frac if args.eval_frac is not None else 0.8

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
        "per_device_train_batch_size" : 1,
        "per_device_eval_batch_size"  : 1,
        "learning_rate"               : 5e-5,
        "remove_unused_columns" : False,
        "label_names"           : [],
    }

    for ea in (ea for ea in extra_args if ea):
        key, value = ea[2:].split('=')
        dictionary = training_config
        if ':' in key: 
            sub, key = key.split(':')
            if not sub in dictionary: dictionary[sub] = {}
            dictionary = training_config[sub]
        dictionary[key] = magic_cast(value)

    setattr(args, 'training_args', transformers.TrainingArguments(**training_config))

    if not os.path.exists(filepath(args.save_dir)): os.makedirs(filepath(args.save_dir), exist_ok=True)

    return args

def magic_cast(v:str):
    if (v.lower()=='true'):  return True
    if (v.lower()=='false'): return False
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: return v

if args is None: 
    args = process_arguments()
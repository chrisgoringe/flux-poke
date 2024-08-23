from argparse import ArgumentParser
import transformers, os
from modules.utils import filepath, SingletonAddin


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

class HintingArguments:
    class MEG:
        def __init__(self, parser:CommentArgumentParser, *args, **kwargs):
            self.meg = parser.add_mutually_exclusive_group(*args, **kwargs)

        def add_argument(self, *args, **kwargs):
            self.meg.add_argument(*args, **kwargs)
            return kwargs.get('default',None)
    
    def __init__(self):
        self.parser = CommentArgumentParser(fromfile_prefix_chars='@')

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        return kwargs.get('default',None)
    
    def add_mutually_exclusive_group(self, *args, **kwargs):
        return HintingArguments.MEG(self.parser, *args, **kwargs)

    def parse_known_args(self, *args, **kwargs):
        args, extra_args = self.parser.parse_known_args(*args, **kwargs)
        for k in args.__dict__: setattr(self, k, getattr(args,k))
        return self, extra_args
    
class Arguments(HintingArguments, SingletonAddin):
    def __init__(self):
        super().__init__()
  
        first = self.add_mutually_exclusive_group(required=True)
        self.first_layer:int    = first.add_argument('--first_layer', help="The first layer in the sub-model.")
        self.first_layers:str   = first.add_argument('--first_layers', help="The range of first layers in the submodel. The whole process runs for each in the range.")

        self.thickness:int              = self.add_argument('--thickness', type=int, default=1, help="The thickness to be trained (default 1)")
        self.model:str                  = self.add_argument('--model', type=str, required=True, help="flux dev model (absolute path)")
        self.load_patches:list[str]     = self.add_argument('--load_patches', action="append", type=str, help="directory to load existing patches from (can have multiple)")
        self.allow_overpatching:bool    = self.add_argument('--allow_overpatching', action="store_true", help="allow one patch to overwrite a previous one (default is an assertion fail)")
        
        self.save_dir:str       = self.add_argument('--save_dir', default="output", help="Relative path of directory to store results in (includes patches if training)")
        self.saved_model:str    = self.add_argument('--saved_model', help="saved model when doing convert (absolute path)")
        self.stats_file:str     = self.add_argument('--stats_file', default="stats.yaml", help="Path for stats to be saved in relative to save_dir")

        self.verbose:bool       = self.add_argument('--verbose', action="store_true")
        self.run_asyncs:bool    = self.add_argument('--run_asyncs', action="store_true", help="Experimental; try to cast asynchronously")
        self.autocast:bool      = self.add_argument('--autocast', action="store_true", help="Use autocast")
        
        self.cast_map:str       = self.add_argument('--cast_map', default=None, help="Relative path to yaml/json file describing how the layers should be cast")
        self.prune_map:str      = self.add_argument('--prune_map', default=None, help="Relative path to yaml/json file describing how the layers should be pruned")
        self.train_map:str      = self.add_argument('--train_map', default=None, help="Relative path to yaml/json file describing how the layers should be trained")
        self.internals:str      = self.add_argument('--internals', default="internals.safetensors", help="Relative path of internals file for prune_map")
        self.default_cast:str   = self.add_argument('--default_cast', default="bfloat16", help="Cast to use for 'default' in the cast_map")

        self.hs_dir:str         = self.add_argument('--hs_dir', default="hidden_states", help="Hugging face repo_id for data")
        self.cache_dir:str      = self.add_argument('--cache_dir', default=None, help="Cache directory for files retrieved from repo")
        self.shuffle:bool       = self.add_argument('--shuffle', action='store_true', help="shuffle the dataset")
        self.shuffle_seed:int   = self.add_argument('--shuffle_seed', default=42)
        self.exclude:list[int]  = self.add_argument('--exclude', action="append", type=int, help="Exclude this folder from the repo (can be used multiple times)")
        
        fraction                = self.add_mutually_exclusive_group()
        self.train_frac:float   = fraction.add_argument('--train_frac', type=float, default=0.8, help="fraction of dataset to train on")
        self.eval_frac:float    = fraction.add_argument('--eval_frac', type=float, default=0.2, help="fraction of dataset to evalaute on")

        act = self.add_mutually_exclusive_group(required=True)
        self.evaluate:bool      = act.add_argument('--evaluate', action='store_true', help='no training, just calculate the loss caused by pruning and/or casting')
        self.train:bool         = act.add_argument('--train', action='store_true', help='training')
        self.convert:bool       = act.add_argument('--convert', action='store_true', help='convert to a standalone model')        
        
        self.training_args:transformers.TrainingArguments = None

        self.process_arguments()

    def process_arguments(self):
    
        _, extra_args = self.parse_known_args([f"@{filepath('arguments.txt')}",])

        def magic_cast(v:str):
            if (v.lower()=='true'):  return True
            if (v.lower()=='false'): return False
            try: return int(v)
            except ValueError: pass
            try: return float(v)
            except ValueError: return v

        training_config = DEFAULT_TRAINING.copy()

        for ea in (ea for ea in extra_args if ea):
            key, value = ea[2:].split('=')
            dictionary = training_config
            if ':' in key: 
                sub, key = key.split(':')
                if not sub in dictionary: dictionary[sub] = {}
                dictionary = training_config[sub]
            dictionary[key] = magic_cast(value)

        self.training_args = transformers.TrainingArguments(**training_config)

        if not os.path.exists(filepath(self.save_dir)): os.makedirs(filepath(self.save_dir), exist_ok=True)

DEFAULT_TRAINING = {
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

args:Arguments = Arguments.instance()

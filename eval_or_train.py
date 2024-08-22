import add_paths

import torch
from safetensors.torch import save_file
import transformers
from modules.arguments import args, filepath
from modules.generated_dataset import TheDataset
from modules.utils import log, shared, int_list_from_string, load_config, is_double
from modules.hffs import HFFS_Cache
from modules.trainer import TheTrainer, prep_for_train
from modules.casting import cast_layer_stack
from modules.pruning import prune_model, apply_patches
from modules.layer import load_single_layer
from modules.sensitivity import Sensitivity
import time
from functools import partial

class TheCallback(transformers.TrainerCallback):
    log:list[dict] = None
    def on_evaluate(self, _, state: transformers.TrainerState, *args, **kwargs):  TheCallback.log = state.log_history.copy()
    def on_train_end(self, _, state: transformers.TrainerState, *args, **kwargs): TheCallback.log = state.log_history.copy()

    @classmethod
    def eval_losses(cls): return [l['eval_loss'] for l in cls.log if 'eval_loss' in l]
    @classmethod
    def losses(cls): return [l['loss'] for l in cls.log if 'loss' in l]

def train_or_evaluate():
    log(f"Model is layers {args.first_layer} to {args.first_layer+args.thickness-1}" if args.thickness > 1 else f"Model is layer {args.first_layer}")
    assert args.first_layer+args.thickness-1 <= shared.last_layer, f"max_layer ({shared.last_layer}) exceeded"

    model = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in range(args.first_layer, args.first_layer+args.thickness)] )
    model.requires_grad_(False)

    if args.prune_map:
        prune_model(model, prune_config=load_config(filepath(args.prune_map)), model_first_layer=args.first_layer, verbose=args.verbose)

    if args.cast_map:
        cast_layer_stack(model, cast_config=load_config(filepath(args.cast_map)), stack_starts_at_layer=args.first_layer, default_cast=args.default_cast, verbose=args.verbose)

    if args.train:
        assert args.train_map, "No train map specified"
        any_to_train = prep_for_train(model, train_config=load_config(filepath(args.train_map)), layer_index=args.first_layer, verbose=args.verbose)
        assert any_to_train, "Nothing to train"
    
    TheDataset.set_dataset_source(dir=args.hs_dir, shuffle=args.shuffle, seed=args.shuffle_seed)
    t = TheTrainer(
        model         = model,
        args          = args.training_args,
        train_dataset = TheDataset(first_layer=args.first_layer, split="train", train_frac=args.train_frac, thickness=args.thickness),
        eval_dataset  = TheDataset(first_layer=args.first_layer, split="eval",  train_frac=args.train_frac, thickness=args.thickness),
        data_collator = transformers.DefaultDataCollator(),
        callbacks     = [TheCallback,],
    )

    start_time = time.monotonic()
    sensitivity = partial(Sensitivity.impact_of_mseloss, args.first_layer+args.thickness, getattr(Sensitivity, args.sensitive_hack))
    if args.evaluate:
        t.evaluate()
        shared.layer_stats[args.first_layer]['loss'] = TheCallback.eval_losses()[0]
        shared.layer_stats[args.first_layer]['sensitivity'] = sensitivity(TheCallback.eval_losses()[0])
    else:
        t.train()
        shared.layer_stats[args.first_layer]['initial_loss'] = TheCallback.eval_losses()[0]
        shared.layer_stats[args.first_layer]['final_loss']   = TheCallback.eval_losses()[-1]
        shared.layer_stats[args.first_layer]['train_loss']   = TheCallback.losses()[-1]
        shared.layer_stats[args.first_layer]['initial_sensitivity'] = sensitivity(TheCallback.eval_losses()[0])
        shared.layer_stats[args.first_layer]['final_sensitivity']   = sensitivity(TheCallback.eval_losses()[-1])
        shared.layer_stats[args.first_layer]['train_sensitivity']   = sensitivity(TheCallback.losses()[-1])
        for i,layer in enumerate(model):
            savefile = filepath(args.save_dir,"{:0>2}.safetensors".format(args.first_layer+i))
            save_file(layer.state_dict(), savefile)
            log(f"Saved in {savefile}")
    shared.layer_stats[args.first_layer]['time'] = time.monotonic() - start_time

    log(str(shared.layer_stats[args.first_layer]))

        
if __name__=="__main__": 
    HFFS_Cache.set_cache_directory(args.cache_dir)
    if args.clear_cache_before: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()

    if args.first_layers=='all':    args.first_layers = f"0-{shared.last_layer}"
    if args.first_layers=='double': args.first_layers = f"{shared.first_double_layer}-{shared.last_double_layer}"
    if args.first_layers=='single': args.first_layers = f"{shared.first_single_layer}-{shared.last_single_layer}"

    shared.set_shared_filepaths(args=args)

    if args.load_patches:
        patched = []
        def record(key): patched.append(key)
        for dir in args.load_patches:
            apply_patches(shared.sd, filepath(dir), [record,])
        assert len(patched)==len(set(patched))

    for l in int_list_from_string(args.first_layers or args.first_layer): 
        args.first_layer = l
        train_or_evaluate()

    shared.save_stats(filepath(args.save_dir,args.stats_file))

    if args.clear_cache_after: 
        log("Clearing hffs cache")
        HFFS_Cache.clear_cache()
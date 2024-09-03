import add_paths
from modules.arguments import args, filepath
from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset, RemoteDataset
from modules.utils import Batcher, shared, is_double, layer_list_from_string
from gguf import GGMLQuantizationType

from modules.jobs import Job
import torch
from tqdm import trange
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from typing import Union
import os

def new_layer(s, n) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    if is_double(n):
        return DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    else:
        return SingleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn)

def load_single_layer(layer_number:int, remove_from_sd=True) -> Union[DoubleStreamBlock, SingleStreamBlock]:
    layer_sd = shared.layer_sd(layer_number)
    if remove_from_sd: shared.drop_layer(layer_number)
    layer = new_layer(None, layer_number)
    layer.load_state_dict(layer_sd)
    return layer

def load_layer_stack():
    print("Loading model")
    layer_stack = torch.nn.Sequential( *[load_single_layer(layer_number=x) for x in trange(shared.last_layer+1)] )
    return layer_stack

def setup():
    HFFS_Cache.set_cache_directory(args.cache_dir)
    shared.set_shared_filepaths(args=args)
    if args.hs_type==2:
        MergedBatchDataset.set_dataset_source(dir=args.hs_dir)
        Batcher.set_mode(all_in_one=True)
    else:
        RemoteDataset.set_dataset_source(dir=args.hs_dir)
        Batcher.set_mode(all_in_one=False)
    Job.layer_generator = new_layer
    Job.args = args
    
def create_dataset():
    if args.hs_type==2:
        return MergedBatchDataset(split='eval', eval_frac=args.eval_frac)
    else:
        return RemoteDataset(split='eval', eval_frac=args.eval_frac, first_layer=0, thickness=57, squeeze=False)

QUANT_FILES = {
    GGMLQuantizationType.Q2_K:'flux1-dev-Q2_K.gguf',
    GGMLQuantizationType.Q3_K:'flux1-dev-Q3_K_S.gguf',
    GGMLQuantizationType.Q4_0:'flux1-dev-Q4_0.gguf',
    GGMLQuantizationType.Q4_1:'flux1-dev-Q4_1.gguf',
    GGMLQuantizationType.Q4_K:'flux1-dev-Q4_K_S.gguf',
    GGMLQuantizationType.Q5_0:'flux1-dev-Q5_0.gguf',
    GGMLQuantizationType.Q5_1:'flux1-dev-Q5_1.gguf',
    GGMLQuantizationType.Q5_K:'flux1-dev-Q5_K_S.gguf',
    GGMLQuantizationType.Q6_K:'flux1-dev-Q6_0.gguf',
    GGMLQuantizationType.Q8_0:'flux1-dev-Q8_0.gguf',
}

QUANT_NAMES = {
    GGMLQuantizationType.Q2_K:'Q2_K*',
    GGMLQuantizationType.Q3_K:'Q3_K_S*',
    GGMLQuantizationType.Q4_0:'Q4_0*',
    GGMLQuantizationType.Q4_1:'Q4_1*',
    GGMLQuantizationType.Q4_K:'Q4_K_S*',    
    GGMLQuantizationType.Q5_0:'Q5_0*',
    GGMLQuantizationType.Q5_1:'Q5_1*',
    GGMLQuantizationType.Q5_K:'Q5_K_S*',
    GGMLQuantizationType.Q6_K:'Q6_0*',
    GGMLQuantizationType.Q8_0:'Q8_0*',
}

def gguf_file(quant:GGMLQuantizationType):
    return os.path.join( args.gguf_dir, QUANT_FILES[quant] )

def get_jobs_list_qpatch(jobs=[]):

    CASTS = [ GGMLQuantizationType.Q5_K, ]
    LAYERS = layer_list_from_string('all')

    for quant in CASTS:
        for layer in LAYERS:
            file = gguf_file(quant)
            config = { 'patches': [{'layers': layer, 'file': file}] }
            label = f"{layer},all,{QUANT_NAMES[quant]}"
            jobs.append( Job(label=label, config=config, preserve_layers=[layer,], ))
    return jobs

def get_jobs_list_prune(jobs=[]):
    LAYERS = layer_list_from_string('all')
    DEPTHS = [2000,4000,6000]

    for layer in LAYERS:
        for depth in DEPTHS:
            config = { "prunes": [{"layers":layer, "remove":depth}]}
            label = f"{layer},all,{depth}"
            jobs.append( Job(label=label, config=config, preserve_layers=[layer,]))

    return jobs

def get_jobs_list_doubles(jobs=[]):
    BLOCKS = [ 'all', ]
    CASTS = ['Linear8bitLt', 'LinearFP4', 'LinearNF4']
    LAYERS = range(10,11)
    
    for cast in CASTS:
        for block in BLOCKS:
            for layer in LAYERS:
                if (False):
                    pass
                else:
                    config = { 'casts': [{'layers': layer, 'blocks': block, 'castto': cast}] }
                    label = f"{layer},{block},{cast}"
                    jobs.append( Job(label=label, config=config, preserve_layers=[layer,],))
    return jobs    

def get_jobs_list_null(jobs=[]) -> list[Job]:
    nzs = []
    def note_nonzero(loss:float, source): 
        if loss>1e-4: 
            print(f"{source} loss {loss}")
            nzs.append(f"{source}")
    def report_nonzero():
        print ("\n".join(nzs))

    jobs.append( Job("null", config={}, preserve_layers=[], callbacks=[note_nonzero,], postrun=report_nonzero))
    return jobs

def get_jobs_list_adding(jobs=[]) -> list[Job]:
    cast = 'Q4_1'
    for first_layer in [4,]:
        for second_layer in [5,14]:
            for first_block in ['img', 'txt']:
                for second_block in ['img', 'txt']:
                    config = { 'casts': [{'layers': first_layer, 'blocks': first_block, 'castto': cast},
                                        {'layers': second_layer, 'blocks': second_block, 'castto': cast}]   }
                    jobs.append( Job(label=f"{first_layer}-{first_block} and {second_layer}-{second_block} to {cast}", 
                                     config=config, 
                                     preserve_layers=[first_layer, second_layer], )
                                )

    return jobs

def main():
    setup()

    jobs:list[Job] = []
    #get_jobs_list_null(jobs)
    #get_jobs_list_adding(jobs)
    #get_jobs_list_qpatch(jobs)
    #get_jobs_list_doubles(jobs)
    get_jobs_list_qpatch(jobs)

    if args.skip: 
        print(f"Skipping {args.skip}")
        jobs = jobs[args.skip:]
    if args.verbose >= 1: print(f"{len(jobs)} jobs")

    the_data    = create_dataset()
    layer_stack = load_layer_stack()

    outfile = os.path.join(args.save_dir, args.results_file)
    if not os.path.exists(os.path.dirname(outfile)): os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open( outfile, 'a+' ) as output_filehandle:
        for i, job in enumerate(jobs):
            result, layer_stack = job.execute(layer_stack, the_data)
            print(f"Job {i} - {result.to_string}")
            print(f"{result.label},{result.loss:>10.5},{result.time:>10.5}", file=output_filehandle, flush=True)

if __name__=='__main__': 
    main()
    shared.save_stats(filepath(args.save_dir,args.stats_file))
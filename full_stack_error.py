import add_paths
from modules.arguments import args, filepath
from modules.hffs import HFFS_Cache
from modules.generated_dataset import MergedBatchDataset, RemoteDataset
from modules.utils import Batcher, shared, layer_list_from_string
from modules.casting import QuantizedTensor
from gguf import GGMLQuantizationType
from modules.precaster import precaster

from modules.jobs import Job
from modules.loader import new_layer, load_layer_stack

import os

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
    QuantizedTensor.NEVER_PURGE = True
    
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
    GGMLQuantizationType.TQ1_0:'TQ1_0',
    GGMLQuantizationType.TQ2_0:'TQ2_0',
    GGMLQuantizationType.Q2_K:'Q2_K*',
    GGMLQuantizationType.Q3_K:'Q3_K_S*',
    GGMLQuantizationType.Q4_0:'Q4_0*',
    GGMLQuantizationType.Q4_1:'Q4_1*',
    GGMLQuantizationType.Q4_K:'Q4_K_S*',    
    GGMLQuantizationType.Q5_0:'Q5_0*',
    GGMLQuantizationType.Q5_1:'Q5_1*',
    GGMLQuantizationType.Q5_K:'Q5_K_S*',
    GGMLQuantizationType.Q6_K:'Q6_K*',
    GGMLQuantizationType.Q8_0:'Q8_0*',
}

def disable_precast(): precaster.enabled = False
def enable_precast():  precaster.enabled = True

def get_jobs_list_cast_all(jobs=[]):
    CASTS = ['QF8_0', 'QF5_0', 'QF4_0' ]
    all_layers = layer_list_from_string('all')
    
    for cast in CASTS:
        config = { 'casts': [{'layers': all_layers, 'castto': cast}] }
        jobs.append( Job(label=cast+" no precast", config=config, preserve_layers=all_layers, prerun=disable_precast))
        jobs.append( Job(label=cast+" precast", config=config, preserve_layers=all_layers, prerun=enable_precast))
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


def main():
    setup()

    jobs:list[Job] = []
    get_jobs_list_null(jobs)
    get_jobs_list_cast_all(jobs)

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
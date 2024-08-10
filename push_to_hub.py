import datasets
import os
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm import tqdm

from huggingface_hub import HfApi


def from_directory_of_files(directory, push_name):
    api = HfApi()
    for file in tqdm(os.listdir(directory)):
        root, ext = os.path.splitext(file)
        if ext=='.safetensors':  
            api.upload_file(
                path_or_fileobj=os.path.join(directory, file),
                path_in_repo=os.path.join(directory, file),
                repo_id=push_name,
            )

    #for file in tqdm(os.listdir(directory)):
    #    root, ext = os.path.splitext(file)
    #    if ext=='.safetensors':  
    #        datum = load_file(os.path.join(directory, file))
    #        ds = datasets.Dataset.from_dict(datum)
    #        ds.push_to_hub(push_name, split=f"p_{root}", num_shards=1)

def from_dataset(directory, push_name):
    ds = datasets.Dataset.load_from_disk(directory)
    ds.push_to_hub(push_name)

if __name__=='__main__':
    a = ArgumentParser()
    a.add_argument('--create', action='store_true')
    a.add_argument('--hidden_states', action='store_true')
    a.add_argument('--prompts', action='store_true')
    args = a.parse_args()

    if args.create:
        api = HfApi()
        api.create_repo(repo_id="ChrisGoringe/fi", repo_type="datset", private=True)
    if args.hidden_states:
        from_directory_of_files(directory="hidden_states", push_name="ChrisGoringe/fi")
    if args.prompts:
        from_dataset(directory="prompts_dataset", push_name="ChrisGoringe/uncleaned_prompts")

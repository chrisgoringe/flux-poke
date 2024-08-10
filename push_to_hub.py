import datasets
import os
from argparse import ArgumentParser
from tqdm import tqdm

from huggingface_hub import HfApi

repo_id = lambda a : f"ChrisGoringe/{a}"

def create_repo(repo):
    api = HfApi()
    api.create_repo(repo_id=repo_id(repo), repo_type="dataset", private=True)

def upload_files_from_directory(repo, directory, max_upload=100):
    api = HfApi()
    for i, file in enumerate(os.listdir(directory)):
        if os.path.splitext(file)[1]=='.safetensors':  
            print(file)
            api.upload_file(
                path_or_fileobj=os.path.join(directory, file),
                path_in_repo=file,
                repo_id=repo_id(repo),
                repo_type="dataset",
            )
        if i==max_upload: return

def upload_dataset(repo, directory):
    ds = datasets.Dataset.load_from_disk(directory)
    ds.push_to_hub(repo_id(repo),)

if __name__=='__main__':
    a = ArgumentParser()
    b = a.add_mutually_exclusive_group(required=True)
    b.add_argument('--create', action='store_true')
    b.add_argument('--hidden_states', action='store_true')
    b.add_argument('--prompts', action='store_true')
    a.add_argument('--repo', required=True)
    args = a.parse_args()

    if args.create:          create_repo(repo=args.repo)
    elif args.hidden_states: upload_files_from_directory(repo=args.repo, directory="hidden_states")
    elif args.prompts:       upload_dataset(repo=args.repo, directory="prompts_dataset")

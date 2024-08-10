import datasets
import os
from safetensors.torch import load_file
from argparse import ArgumentParser

def from_directory_of_files(directory, push_name):
    def gen():
        for file in os.listdir(directory):
            root, ext = os.path.splitext(file)
            if ext=='.safetensors':  
                datum = load_file(os.path.join(directory, file))
                datum['id'] = root
                yield datum

    ds = datasets.Dataset.from_generator(gen)
    ds.push_to_hub(push_name)

def from_dataset(directory, push_name):
    ds = datasets.Dataset.load_from_disk(directory)
    ds.push_to_hub(push_name)

if __name__=='__main__':
    a = ArgumentParser()
    a.add_argument('--hidden_states', action='store_true')
    a.add_argument('--prompts', action='store_true')
    a.parse_args()

    if a.hidden_states:
        from_dataset(directory="hidden_states", push_name="ChrisGoringe/flux_internals")
    if a.prompts:
        from_directory_of_files(directory="prompts_dataset", push_name="ChrisGoringe/uncleaned_prompts")

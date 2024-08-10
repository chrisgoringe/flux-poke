import datasets
import os
from safetensors.torch import load_file

def from_directory_of_files(directory, push_name):
    def gen():
        for file in os.listdir(directory):
            root, ext = os.path.splitext(file)
            if root=='.safetensors':  
                datum = load_file(os.path.join(directory, file))
                datum['id'] = root
                yield datum

    ds = datasets.Dataset.from_generator(gen)
    ds.push_to_hub(push_name)

def from_dataset(directory, push_name):
    ds = datasets.Dataset.load_from_disk(directory)
    ds.push_to_hub(push_name)

HF_NAME, DIRECTORY, ALREADY_DATASET = ("ChrisGoringe/flux_internals","hidden_states",False)
#HF_NAME, DIRECTORY, ALREADY_DATASET = ("ChrisGoringe/uncleaned_prompts","prompts_dataset",True)

if __name__=='__main__':
    if ALREADY_DATASET:
        from_dataset(directory=DIRECTORY, push_name=HF_NAME)
    else:
        from_directory_of_files(directory=DIRECTORY, push_name=HF_NAME)

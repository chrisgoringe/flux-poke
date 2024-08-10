import datasets
import os
from safetensors.torch import load_file

def main(directory, push_name):
    def gen():
        for file in os.listdir(directory):
            root, ext = os.path.splitext(file)
            if root=='.safetensors':  
                datum = load_file(os.path.join(directory, file))
                datum['id'] = root
                yield datum

    ds = datasets.Dataset.from_generator(gen)
    ds.push_to_hub(push_name)

HF_NAME = "ChrisGoringe/flux_internals"
DIRECTORY = "hidden_states"

if __name__=='__main__':
    main()

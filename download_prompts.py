from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser

# Edit the line below to read 
# option = 0  (or whatever )

OPTIONS = [
    ("isidentical/random-stable-diffusion-prompts", 'prompt'),  #0
    ("Gustavosta/Stable-Diffusion-Prompts", 'Prompt'),          
]

if __name__=='__main__':
    a = ArgumentParser()
    a.add_argument('--option', default=0)
    args = a.parse_args()
    filename, column_name = OPTIONS[args.option]
    dataset = load_dataset(filename)

    with open("prompts.txt","w", encoding="UTF-8") as f:
        for item in tqdm(dataset['train']):
            print(item[column_name], file=f, end='')
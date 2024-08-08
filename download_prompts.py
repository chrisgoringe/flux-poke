from datasets import load_dataset
from tqdm import tqdm

# Edit the line below to read 
# option = 0  (or whatever )
option = None
OPTIONS = [
    ("isidentical/random-stable-diffusion-prompts", 'prompt'),  #0
    ("Gustavosta/Stable-Diffusion-Prompts", 'Prompt'),          

]


if __name__=='__main__':
    assert option is not None, "edit this file to choose an option"    
    filename, column_name = OPTIONS[option]
    dataset = load_dataset(filename)

    with open("prompts.txt","w", encoding="UTF-8") as f:
        for item in tqdm(dataset['train']):
            print(item[column_name], file=f, end='')
from datasets import load_dataset
from tqdm import tqdm
dataset = load_dataset("isidentical/random-stable-diffusion-prompts")

with open("prompts.txt","w", encoding="UTF-8") as f:
    for item in tqdm(dataset['train']):
        print(item['prompt'], file=f, end='')

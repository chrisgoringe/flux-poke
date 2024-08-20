
import requests, datasets
from tqdm import trange, tqdm

def prompts(period="Week"):
    for i in trange(1,6):
        r = requests.get(url="https://civitai.com/api/v1/images", params={
                            "limit":200,
                            "modelVersionId":691639,
                            "period":period,"page":i})
        j = r.json()
        if 'items' in j:
            for item in tqdm(r.json()['items']):
                if (prompt:=(item.get('meta',None) or {}).get('prompt',None)): yield prompt.replace("\n"," ")
        else:
            print(f"Getting {i}: {j}")
            break

def main(add=True):
    dataset = datasets.Dataset.load_from_disk('prompts_dataset')
    print(len(dataset))
    if add:
        for prompt in prompts():             dataset = dataset.add_item({"prompt":prompt})
        for prompt in prompts(period="Day"): dataset = dataset.add_item({"prompt":prompt})
        print(len(dataset))
        dataset = datasets.Dataset.from_dict( {'prompt':dataset.unique('prompt')} )
        print(len(dataset))
        dataset.save_to_disk('new_prompts_dataset')

if __name__=='__main__': 
    main()
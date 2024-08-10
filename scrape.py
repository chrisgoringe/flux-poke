
import time, requests
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

def scrape():
    with open('prompts.txt','w', encoding='UTF-8') as f:
        for prompt in prompts(): print(prompt, file=f)
        for prompt in prompts(period="Day"): print(prompt, file=f)

def unique():
    with open('prompts.txt','r', encoding='UTF-8') as f:
        s = set()
        for prompt in f.readlines(): s.add(prompt)
    with open('prompts.txt','w', encoding='UTF-8') as f:
        for prompt in s: print(prompt, file=f, end="") 
    
if __name__=='__main__': 
    scrape()
    unique()
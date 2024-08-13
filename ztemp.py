from huggingface_hub import HfFileSystem
from huggingface_hub.utils._errors import HfHubHTTPError
import time, regex

repo_id = "ChrisGoringe/fi"  
fs = HfFileSystem()        

incomplete = True
re = regex.compile("datasets/ChrisGoringe/fi/[0-9_]*.safetensors")

to_do = len(fs.glob("/".join(("datasets",repo_id,"*"))))
while incomplete:
    done = 0
    try:
        for path in fs.glob("/".join(("datasets",repo_id,"*"))): 
            if re.match(path): fs.rm(path)
            done += 1
            print(f"\r{done}/{to_do}",end='')
        incomplete = False
    except HfHubHTTPError:
        print("\nErrored")
        time.sleep(30)
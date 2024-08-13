from huggingface_hub import HfFileSystem
from huggingface_hub.utils._errors import HfHubHTTPError
import time, regex

repo_id = "ChrisGoringe/fi"  
fs = HfFileSystem()        

incomplete = True
re = regex.compile("datasets/ChrisGoringe/fi/[0-9_]*.safetensors")

while incomplete:
    try:
        for path in fs.glob("/".join(("datasets",repo_id,"*"))): 
            if re.match(path): fs.rm(path)
        incomplete = False
    except HfHubHTTPError:
        time.sleep(30)
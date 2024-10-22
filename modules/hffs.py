from huggingface_hub import HfFileSystem
from huggingface_hub.utils._errors import HfHubHTTPError
from safetensors.torch import load_file, save_file
import tempfile, os, torch, sys, random, tqdm
from .utils import SingletonAddin, Batcher

class HFFS_Cache(SingletonAddin):
    def __init__(self):
        self._directory = None

    @property
    def directory(self):
        if self._directory is None: 
            self._temp_directory = tempfile.TemporaryDirectory()
            self._directory = self._temp_directory.name
        return self._directory
    
    @classmethod
    def set_cache_directory(cls, directory):
        cls.instance()._directory = directory
        if not os.path.exists(directory): os.makedirs(directory, exist_ok=True)

    def clear_cache(self):
        for f in os.listdir(self._directory): os.remove(os.path.join(self._directory, f))

    def is_in_cache(self, filename):
        return os.path.exists(self.localname(filename))

    def get_from_cache(self, filename):
        return load_file(self.localname(filename))
    
    def store_in_cache(self, filename, data):
        save_file(data, self.localname(filename))
    
    def localname(self, filename:str):
        a, b = filename.split("/")[-2:]
        if not os.path.exists(subdir:=os.path.join(self.directory,a)): os.makedirs(subdir,exist_ok=True)
        return os.path.join(subdir, b+".safetensors")
    
    def tempname(self):
        return os.path.join(self.directory, "temp.safetensors")

class HFFS:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        self.fs = HfFileSystem()
        self.cache = HFFS_Cache.instance()

    def set_repo_id(self, repo_id):
        self.repo_id = repo_id

    def rpath(self, filename):
        return "/".join(["datasets",self.repo_id, filename])

    def get_entry_list(self, validate=False) -> list[str]:
        entries = self.fs.glob(self.rpath('[0-9]*/'))
        if validate:
            valid = []
            print("Validating dataset...")
            for e in tqdm.tqdm(entries):
                if len(self.fs.glob(f"{e}/*.safetensors"))==9: valid.append(e)
                else: print(f"{e.split('/')[-1]} not valid")
            return valid
        return entries
    
    def load_file(self, filename):
        def convert(filename):
            f = filename.split("/")
            try: n = int(f[-1])
            except: n = None
            return Batcher.filename("/".join(f[:-1]), n)
        filename = convert(filename)
        #print(f"Loading {filename}")
        if self.cache.is_in_cache(filename): 
            #print("From cache")
            return self.cache.get_from_cache(filename)
        with tempfile.TemporaryDirectory() as tempdir:
            tempname = os.path.join(tempdir,str(random.randint(1,1000000))+".safetensors")
            #print("Downloading")
            self.fs.get_file(rpath=filename, lpath=tempname)
            data = load_file(tempname)
            self.cache.store_in_cache(filename, data)
        return data
    
    def save_file(self, label:str, datum:dict[str,torch.Tensor]) -> bool:
        with tempfile.TemporaryDirectory() as tempdir:
            tempname = os.path.join(tempdir,str(random.randint(1,1000000)))
            save_file(datum, tempname)
            try:
                self.fs.put_file(lpath=tempname, rpath=self.rpath(label))
                return True
            except HfHubHTTPError:
                print(sys.exc_info())
                return False

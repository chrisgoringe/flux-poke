from huggingface_hub import HfFileSystem
from huggingface_hub.utils._errors import HfHubHTTPError
from safetensors.torch import load_file, save_file
import tempfile, os, torch
from .utils import SingletonAddin

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
    
    def localname(self, filename):
        return os.path.join(self.directory, os.path.split(filename)[-1])
    
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

    def get_entry_list(self) -> list[str]:
        return self.fs.glob(self.rpath(""))
    
    def load_file(self, filename):
        print(f"Loading {filename}")
        if self.cache.is_in_cache(filename): 
            print("From cache")
            return self.cache.get_from_cache(filename)
        with tempfile.NamedTemporaryFile() as tempname:
            print("Downloading")
            self.fs.get_file(rpath=filename, lpath=tempname)
            data = load_file(tempname)
            self.cache.store_in_cache(filename, data)
        return data
    
    def save_file(self, label:str, datum:dict[str,torch.Tensor]) -> bool:
        with tempfile.NamedTemporaryFile() as tempname:
            save_file(datum, tempname)
            try:
                self.fs.put_file(lpath=tempname, rpath=self.rpath(label))
                return True
            except HfHubHTTPError:
                return False

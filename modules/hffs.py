from huggingface_hub import HfFileSystem
from safetensors.torch import load_file, save_file
import tempfile, os
from .utils import SingletonAddin
from threading import Lock

class HFFS_Cache(SingletonAddin):
    def __init__(self):
        self._directory = None
        self.templock = Lock()

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

    def get_file_list(self, pattern="*.safetensors") -> list[str]:
        return self.fs.glob("/".join([self.repo_id, pattern]))
    
    def load_file(self, filename, filter:callable=lambda a:a):
        print(f"Loading {filename}")
        if self.cache.is_in_cache(filename): 
            print("From cache")
            return self.cache.get_from_cache(filename)
        with self.cache.templock:
            tempname = self.cache.tempname()
            print("Downloading")
            self.fs.get_file(rpath=filename, lpath=tempname)
            data = filter(load_file(tempname))
            os.remove(tempname)
            self.cache.store_in_cache(filename, data)
        return data
    
    def save_file(self, label, datum):
        with self.cache.templock:
            tempname = self.cache.tempname()
            save_file(datum, tempname)
            self.fs.put_file(lpath=tempname, rpath=label)
            os.remove(tempname)

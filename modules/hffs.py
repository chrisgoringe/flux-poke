from huggingface_hub import HfFileSystem
from safetensors.torch import load_file, save_file
import tempfile, os
from utils import SingletonAddin
from threading import Lock

class HFFS_Cache(SingletonAddin):
    def __init__(self):
        self._directory = None
        self.templock = Lock()

    @property
    def directory(self):
        if self._directory is None: self._directory = tempfile.TemporaryDirectory()
        return self._directory
    
    @classmethod
    def set_cache_directory(cls, directory):
        cls.instance()._directory = directory

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
    
    def set_validity_token(self, token:str):
        '''
        A string that somehow describes what has been cached.
        If it is different from that stored in the cache directory, the cache is invalid and should be cleared.
        '''
        vt_file = self.localname('validity_token.txt')
        if os.path.exists(vt_file):
            with open(vt_file, 'r') as f: vt = f.readline()
            if vt.strip() != token.strip(): self.clear_cache()
        with open(vt_file, 'w') as f: print(token, file=f)

class HFFS:
    def __init__(self, repository="datasets/ChrisGoringe/fi"):
        self.repository = repository
        self.fs = HfFileSystem()

    def get_file_list(self, pattern="*.safetensors") -> list[str]:
        return self.fs.glob("/".join([self.repository, pattern]))
    
    def load_file(self, filename, filter:callable=lambda a:a):
        cache = HFFS_Cache.instance()
        if cache.is_in_cache(filename): return cache.get_from_cache(filename)
        with cache.templock:
            tempname = cache.tempname()
            self.fs.get_file(rpath=filename, lpath=tempname)
            data = filter(load_file(tempname))
            cache.store_in_cache(filename, data)
        return data
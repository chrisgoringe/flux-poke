import logging, time, yaml
from safetensors.torch import load_file
from modules.arguments import args, filepath

class SingletonAddin:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

class Log(SingletonAddin):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start = time.monotonic()

    def info(self, msg):
        self.logger.info("{:>8.1f}s ".format(time.monotonic()-self.start)+msg)

log = Log.instance().info

class Shared(SingletonAddin):
    def __init__(self):
        self.sd        = load_file(args.model)
        self.internals = load_file(filepath(args.internals))
        self.max_layer = 18
        self.layer_stats = [{"layer":i} for i in range(19)]

    @property
    def layer_stats_yaml(self):
        return yaml.dump( { f"layer{i}":layer_stats for i, layer_stats in enumerate(self.layer_stats) } )

    
shared = Shared.instance()

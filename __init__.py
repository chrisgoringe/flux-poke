from .trackers import FluxWatcher, FluxInternalsSaver, FluxMasker
from .prompts import Prompts, Counter

NODE_CLASS_MAPPINGS = { 
    "Insert Probes" : FluxWatcher,
    "Save Internals" : FluxInternalsSaver,
    "Internal Mask" : FluxMasker,
    "Prompts" : Prompts,
    "Counter" : Counter,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]
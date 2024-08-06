from .flux_probe_nodes import InsertInternalProbes, InternalsSaver, InsertHiddenStateProbes, HiddenStatesSaver, LoadPrunedFluxModel, ReplaceLayers
from .utils import Prompts, Counter

NODE_CLASS_MAPPINGS = { 
    "Insert Internal Probes" : InsertInternalProbes,
    "Save Internal Data" : InternalsSaver,
    "Insert Hidden State Probes" : InsertHiddenStateProbes,
    "Save Hidden State Data" : HiddenStatesSaver,
    "Load Pruned Model" : LoadPrunedFluxModel,
    "Replace Layers" : ReplaceLayers,
    "Prompts" : Prompts,
    "Counter" : Counter,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]
from .flux_probe_nodes import InsertInternalProbes, InternalsSaver, InsertHiddenStateProbes, HiddenStatesSaver, \
    LoadPrunedFluxModel, ReplaceLayers, LoadPrunedFluxModelThreshold
from .utils_nodes import Prompts, Counter, FluxSimpleScheduler, RandomSize

NODE_CLASS_MAPPINGS = { 
    "Insert Internal Probes"        : InsertInternalProbes,
    "Save Internal Data"            : InternalsSaver,
    "Insert Hidden State Probes"    : InsertHiddenStateProbes,
    "Save Hidden State Data"        : HiddenStatesSaver,
    "Load Pruned Model"             : LoadPrunedFluxModel,
    "Load Pruned Model (Threshold)" : LoadPrunedFluxModelThreshold,
    "Load Patched Model"            : ReplaceLayers,
    "Flux Simple Scheduler"         : FluxSimpleScheduler,
    "Prompts"                       : Prompts,
    "Counter"                       : Counter,
    "Random Size"                   : RandomSize,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]
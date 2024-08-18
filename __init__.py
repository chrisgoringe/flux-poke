from .flux_probe_nodes import InsertProbes, SaveProbeData,  \
    LoadPrunedFluxModel, ReplaceLayers, LoadPrunedFluxModelThreshold
from .utils_nodes import Prompts, Counter, FluxSimpleScheduler, RandomSize, CommonSizes, QPause, RandomInt
from .mixed_casting_node import UnetLoaderMixed
#from .cpu_offload_node import CPUOffLoad

NODE_CLASS_MAPPINGS = { 
    "Insert Probes"                 : InsertProbes,
    "Save Probe Data"               : SaveProbeData,

    "Load Pruned Model"             : LoadPrunedFluxModel,
    "Load Pruned Model (Threshold)" : LoadPrunedFluxModelThreshold,
    "Load Patched Model"            : ReplaceLayers,

    "Load Mixed Cast Model"         : UnetLoaderMixed,
    #"CPU Offload Layers"            : CPUOffLoad,

    "Random Int"                    : RandomInt,
    "Flux Simple Scheduler"         : FluxSimpleScheduler,
    "Prompts"                       : Prompts,
    "Counter"                       : Counter,
    "Random Size"                   : RandomSize,
    "Common Flux Sizes"             : CommonSizes,
    "Queue Limit Pause"             : QPause,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]
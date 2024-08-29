from .add_paths import paths_added
from .nodes_flux_probe_nodes import InsertProbes, SaveProbeData,  \
    LoadPrunedFluxModel, ReplaceLayers, LoadPrunedFluxModelThreshold
from .nodes_utils_nodes import Prompts, Counter, FluxSimpleScheduler, RandomSize, CommonSizes, QPause, RandomInt
from .nodes_mixed_casting_node import UnetLoaderMixed



NODE_CLASS_MAPPINGS = { 
    "Insert Probes"                 : InsertProbes,
    "Save Probe Data"               : SaveProbeData,

    #"Load Pruned Model"             : LoadPrunedFluxModel,
    #"Load Pruned Model (Threshold)" : LoadPrunedFluxModelThreshold,
    #"Load Patched Model"            : ReplaceLayers,

    #"Load Mixed Cast Model"         : UnetLoaderMixed,

    "Random Int"                    : RandomInt,
    "Flux Simple Scheduler"         : FluxSimpleScheduler,
    "Prompts"                       : Prompts,
    "Counter"                       : Counter,
    "Random Size"                   : RandomSize,
    "Common Flux Sizes"             : CommonSizes,
    "Queue Limit Pause"             : QPause,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]
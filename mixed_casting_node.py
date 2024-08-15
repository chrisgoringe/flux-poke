import folder_paths
import comfy
from safetensors.torch import load_file
import torch
import json

class MixedOps:
    castings = None
    @classmethod
    def load_castings(cls, filepath):
        with open(filepath) as f: cls.castings = json.load(f)

    class Linear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight = None
            self.bias   = None
            self.parameters_manual_cast = torch.float32

        def forward(self, x):
            with torch.autocast("cuda"): return super(x)

        def type_from_prefix(self, prefix:str):
            for k in MixedOps.castings:
                if k in prefix: return getattr(torch, MixedOps.castings[k])
            return None

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            self.my_type = self.type_from_prefix(prefix)
            for k,v in state_dict.items():
                if k[len(prefix):] == "weight":
                    self.weight = v.to(self.my_type)  # This will also handle size if we're loading something saved pruned...
                elif k[len(prefix):] == "bias":
                    self.bias = v.to(self.my_type)
                else:
                    missing_keys.append(k)  # wrong, surely?

        def _apply(self, fn):
            if self.weight is not None:
                self.weight = fn(self.weight)
            if self.bias is not None:
                self.bias = fn(self.bias)
            super()._apply(fn)
            return self

class UnetLoaderMixed:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet") if x.endswith(".gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "casting_file": ("STRING", {"default":""}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, casting_file):
        MixedOps.load_castings(casting_file)
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = load_file(unet_path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": MixedOps}
        )
        return (model,)
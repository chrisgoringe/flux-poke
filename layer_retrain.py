import sys, os
sys.path.insert(0,os.getcwd())

from trackers import slice_double_block
from comfy.ldm.flux.layers import DoubleStreamBlock
import torch
from safetensors.torch import load_file, save_file
import transformers
from functools import partial

import logging
logger = logging.getLogger(__name__)

filepath = partial(os.path.join,os.path.split(__file__)[0])

class Shared:
    sd = load_file("D:/models/unet/flux1-dev.sft")
    internals = load_file(filepath("internals.safetensors"))

class TheTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()

    def compute_loss(self, model:DoubleStreamBlock, inputs:dict[str,torch.Tensor], return_outputs=False):
        img_out, txt_out = model( inputs['img'], inputs['txt'], inputs['vec'], inputs['pe'] )
        loss = self.loss_fn(torch.cat((img_out,txt_out),dim=1), torch.cat((inputs['img'],inputs['txt']), dim=1))
        return (loss, (img_out, txt_out)) if return_outputs else loss
    
class TheDataset:
    def __init__(self, layer:int, split:str, train_frac=0.8):
        dir = filepath("hidden_states")
        self.sources = [ os.path.join(dir,x) for x in os.listdir(dir) if x.endswith(".safetensors") ]
        split_at = int(train_frac*len(self.sources))
        if   split=='train': self.sources = self.sources[:split_at]
        elif split=='eval':  self.sources = self.sources[split_at:]
        self.layer = layer
    
    def __len__(self): 
        return len(self.sources)

    def __getitem__(self, i):
        all_data = load_file(self.sources[i])
        return {
            "img"     : all_data[f"{self.layer}-img"].squeeze(0),
            "txt"     : all_data[f"{self.layer}-txt"].squeeze(0),
            "vec"     : all_data[f"{self.layer}-vec"].squeeze(0),
            "pe"      : all_data[f"{self.layer}-pe"].squeeze(0),
            "img_out" : all_data[f"{self.layer+1}-img"].squeeze(0),
            "txt_out" : all_data[f"{self.layer+1}-txt"].squeeze(0),
        }

def load_layer(layer_number:int) -> DoubleStreamBlock:
    layer = DoubleStreamBlock(hidden_size=3072, num_heads=24, mlp_ratio=4, dtype=torch.bfloat16, device="cpu", operations=torch.nn, qkv_bias=True)
    prefix = f"double_blocks.{layer_number}."
    layer_sd = { k[len(prefix):]:Shared.sd[k] for k in Shared.sd if k.startswith(prefix) }
    layer.load_state_dict(layer_sd)
    return layer

def main():
    # Load the model
    LAYER = 10
    model = load_layer(layer_number=LAYER)
    logger.info(f"Loaded  img as {model.img_mlp[0].out_features} and txt as {model.txt_mlp[0].out_features}")

    # Prune the model
    THRESHOLD = 20000
    img_data = Shared.internals[f"double-img-{LAYER}"]
    txt_data = Shared.internals[f"double-txt-{LAYER}"]
    def mask_from(data): return list( d>THRESHOLD for d in data )
    slice_double_block(model, mask_from(img_data), mask_from(txt_data))
    logger.info(f"Reduced img to {model.img_mlp[0].out_features} and txt to {model.txt_mlp[0].out_features}")

    # Load the dataset
    training_dataset = TheDataset(layer=LAYER, split="train")
    eval_dataset     = TheDataset(layer=LAYER, split="eval")

    # Set up training
    training_config = {
        "save_strategy" : "no",
        "eval_strategy" : "steps",
        "logging_strategy" : "steps",
        "output_dir" : "output",
        #"gradient_checkpointing":True,
        "max_steps":100,
        "logging_steps":10,
        "eval_steps":100,
        "per_device_eval_batch_size":1,
        "remove_unused_columns":False,
        "label_names":[],
        "eval_on_start":True,
        "lr_scheduler_type":"cosine",
    }
    
    t = TheTrainer(
        model         = model,
        args          = transformers.TrainingArguments(**training_config),
        train_dataset = training_dataset,
        eval_dataset  = eval_dataset,
        data_collator = transformers.DefaultDataCollator(),
    )
    t.can_return_loss = True
    t.train()
    save_file(model.state_dict(), filepath("retrained_layers",f"{LAYER}.safetensors"))


if __name__=="__main__": main()
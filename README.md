# flux-poke

An effort to make the Flux-dev model run in less memory...

## Step by step

### Gather analysis data

- Enable the `Enable to run with analysis` block, but then bypass the two nodes referring to `Hidden State` (unless you are happy to spend a lot of disc space on data!)
- Use the prompt generator, or put your own prompts into CLIPTextEncodeFlux
- Mess with the settings all you like
- Each time you run, internal statistics will be saved into `internals.safetensors`
 - These stats record how often each internal state (there are 466,944 of them) gets activated (out of typically 80,000 possible triggers per image)
- The more runs, the better the stats will be

### Play with a pruned version

- Bypass `Enable to run with analysis` and enable the `Enable to run with pruned model` block.
- Try generating images with the model loaded using one of the Pruned nodes.
- You can choose different ranges of layers (they are numbered 0-18 inclusive)
- Either set a number of lines to cut ()







# Notes

## DoubleStreamBlocks (19) - 2.8B mlp parameters, 2.15B attn-mod parameters, 1.4B self-attn parameters
 
Flux has 19 `DoubleStreamBlock`s, each of which has two `mlp` (`img` and `txt`), each consisting of two Linear matrices. These have internal sizes of 12288 and hidden_size of 3072, so a total of 151M parameters each layer (12288.3072.2.2), or 2.8B parameters.

img_mod and txt_mod are 3072x18432 each - 2.15B

attn is 3072x3072x4 for each of txt and img - 1.4B

## SingleStreamBlocks (38) - 2.8B mlp parameters, 2.15B attn-mod parameters, no self attn

The `SingleStreamBlock`s (of which there are 38) have  
linear1 (in_features=3072, out_features=21504, bias=True) and 
linear2 (in_features=15360, out_features=3072, bias=True)

linear1 maps the hidden state (3072) to (qkvi), where i is the 12288 intermediate states

linear2 maps (ai) where a is the post-attention value of qkv onto hidden state.

So each of this includes a submatrix, linear1[9216:,:] and linear2[:,3072:] which are mlp-like. In these layers both the img and the txt data streams are multiplied by the same matrices.

## Download prompts

Optionally, run `python download_prompts.py` in the custom node's directory to download prompts

## Workflow

### Normal Run

### Analysis

Updates `internals.safetensors` and adds to the `hidden_states` directory

Done for 100 images from the whole dataset, then another 100 requiring there to be a `"` in the prompt

### Pruned Run

Just throw stuff away

### layer retrain

Uses `internals.safetensors` and the `hidden_states` directory. Prunes a layer, retrains it,
and saves it in `retrained_layers`

### Replaced run

Loads the model, replaces the specified layers with retrained versions
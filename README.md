# flux-poke
 
Flux has 19 `DoubleStreamBlock`s, each of which has two `mlp` (`img` and `txt`), each consisting of two Linear matrices.

## Download prompts

Optionally, run `python download_prompts.py` in the custom node's directory to download 3.3 million prompts

## Workflow

### Normal Run

### Analysis

Updates `internals.safetensors` and adds to the `hidden_states` directory

### Pruned Run

Just throw stuff away

### layer retrain

Uses `internals.safetensors` and the `hidden_states` directory. Prunes a layer, retrains it,
and saves it in `retrained_layers`

### Replaced run

Loads the model, replaces the specified layers with retrained versions
# Making a smaller Flux

## Analysis

- Generate data
    - internals.safetensors tracking internal state activation
    - ChrisGoringe/fi with hidden state data
- Test losses by pruning
    - img_count = txt_count = x_count = 4000  - evaluiate losses

- Test losses by casting
    - e4m3fn best of the 8bits



## Memory Factor

|Elements|Percentage of Model|Per Layer|
|-|-|-|
|Double Block Layers|54.26|2.86|
|...Double Block mlps|24.11|1.27|
|Single Block Layers|45.22|2.86|
|...Single Block l1,l2|36.17|0.95|
|In and Final|0.54|0.54|



## Process

Prune and retrain 
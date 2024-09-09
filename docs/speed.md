# Speed

For full stack cast, 120 'steps'. Speed measured in steps per second (step = full pass through all 57 layers)

Running on A40, pytorch 2.2.0, cuda 12.1.1, transformers 4.44.2, gguf 0.10.0

|cast|precast|speed|
|-|-|-|
|none|n/a|1.273|
|Q8_0| no|1.074|
|Q8_0|yes|
|Q5_1|no|0.568|
|Q4_1|no|0.847|

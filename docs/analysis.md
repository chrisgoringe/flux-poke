# Analysis

## Casting

See [sensitivity analysis](./Sensitivity.md) for details.

- layers 0-2
    - leave in bfloat16
- layers 3-9
    - Q8_0
- layers 10-12
    - Q5_1 for img, Q8_0 for txt
- layers 13-17
    - Q5_1
- layer 18
    - leave in bfloat16
- layers 19-56 (all single)
    - Q4_1

### Proposed regime

```
NOP = no prune, no train
P4  = prune 4k lines
P4T = prune 4k lines and retrain (1000 steps)
P2T = prune 2k lines and retrain (1000 steps)

- img
    -  0- 2 NOP
    -  3-17 P4T
    -    18 NOP
- txt
    -  0-17 P4
    -    18 NOP
- x
    - 19-   P4T ? P2T
```

### Then... look at casting errors
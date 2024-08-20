# Analysis

## Prune and train

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
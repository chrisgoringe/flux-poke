import os, yaml


def path(f):
    return os.path.join(os.path.dirname(__file__), f)

def from_csv():
    costs = { layer:{} for layer in range(57) }

    with open(path("casting results.csv"),'r') as f:
        for line in f.readlines():
            bits = [b.strip() for b in line.split(",")]
            try:
                if bits[1]=='all':
                    layer = int(bits[0])
                    cast = bits[2]
                    cost = float(bits[3])
                    costs[layer][cast] = cost
            except:
                pass

    return costs

def to_yaml(costs):
    with open(path("casting_cost.yaml"), 'w') as f:
        print(yaml.dump(costs), file=f)

def from_yaml():
    with open(path("casting_cost.yaml"), 'r') as f:
        return yaml.safe_load(f)

PRELUDE = '''# Casting cost

## General Observations

- bitsandbytes doesn't perform well
- mostly the on the fly ones work well, although Qx_K_S perform better or as well and are slightly smaller
- there is a very strong dependance on depth in the model, though not entirely monotonic
- some interesting variation in last 10 or so layers with smaller quants performing very badly
- the quantisation or not of bias (the difference between Q4_1 and Q4_1* and similar) seems to make no difference
- Q6_K is really good

Model - Flux.1.dev

Quantizations marked with (*) are patched in from GGUF models. Others are
quantised by this code.

In all models, the entry and exit layers are unquantised: 64,124,992 parameters

In all models, the normalisation scales are unquantised: 19,456 parameters

In patch models, block biases are unquantised: 3,035,136 parameters

|type|quantised|unquantised block biases|unquantised other|q%|
|:-:|-:|-:|-:|-:|
|fly|11,837,294,656|0|64,144,448|99.461%|
|patch|11,834,228,736|3,065,920|64,144,448|99.435%|


---

## Error (average MSE error in final hidden state) of quantising layers to different levels.

Note that layers 19-56 are single block (141,557,760 quantable parameters), 
layers 0-18 are double block (339,738,624 quantable parameters). Quantizing a double
block saves 2.4 times the memory of quantizing a single block. See weighted table below
to make comparisons between layers.

'''

MIDDLE = '''
---

## Weighted by parameter count

Same again, but the double block values divided by 2.4. So this is proportional to error per parameter quantized.

'''

POSTLUDE = '''
Patches from (https://huggingface.co/city96/FLUX.1-dev-gguf)
'''

def to_md(costs):
    #all_casts = ['Q8_0', 'bf8', 'bnb8', 'Q5_1', 'Q5_K_S*', 'Q4_1', 'Q4_1*', 'Q4_K_S*', 'Q4_0*', 'bnbFP4', 'bnbNF4', 'Q3_K_S*', 'Q2_K*']
    casts_and_bits = {
        'Q8_0':8.5,
        'bf8':8,
        'Q6_K*':6.5625,
        'Q5_1':6,
        'Q5_K_S*':5.5, 
        'Q4_1':5, 
        'Q4_1*':5, 
        'Q4_K_S*':4.5, 
        'Q4_0*':4.5, 
        'Q3_K_S*':3.4375,
        'Q2_K*':2.625
    }
    #for layer in costs.values():
    #    for cast in layer:
    #        if not cast in all_casts: all_casts.append(cast)

    def format(x, divide_by=1.0):
        if not x: return ""
        return f"{x/divide_by: >7.3f}"

    with open(path("casting_cost.md"), 'w') as f:
        print(PRELUDE, file=f)

        print( "|-|" + "|".join(casts_and_bits) + "|", file=f )
        print( "|-|" + "|".join("-:" for _ in casts_and_bits) + "|", file=f )
        print("|bits|" + "|".join(str(casts_and_bits[x]) for x in casts_and_bits) + "|", file=f )
        for layer in costs:
            print( f"|{layer}|" + "|".join( format(costs[layer].get(cast,None)) for cast in casts_and_bits ) + "|", file=f)

        print(MIDDLE, file=f)

        print( "|-|" + "|".join(casts_and_bits) + "|", file=f )
        print( "|-|" + "|".join("-:" for _ in casts_and_bits) + "|", file=f )
        print("|bits|" + "|".join(str(casts_and_bits[x]) for x in casts_and_bits) + "|", file=f )
        for layer in costs:
            print( f"|{layer}|" + "|".join( format(costs[layer].get(cast,None), (2.4 if int(layer)<19 else 1.0)) for cast in casts_and_bits ) + "|", file=f)


        print(POSTLUDE, file=f)

        
to_yaml(from_csv())
to_md(from_yaml())
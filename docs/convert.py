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

## Bits per parameter:

|-|Q8_0|bf8|bnb8|Q5_K_S*|Q5_1|Q4_0*|Q4_1|Q4_1*|Q4_K_S*|bnbFP4|bnbNF4|Q3_K_S*|Q2_K*|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|bits|8.5|8|8+|5.5|6|6.5625|5|5|4.5|4+|4+|3.4375|2.625|

---

## Error (average MSE error in final hidden state) of quantising layers to different levels.

Note that layers 19-56 are single block (141,557,760 quantable parameters), 
layers 0-18 are double block (339,738,624 quantable parameters). Quantizing a double
block saves 2.4 times the memory of quantizing a single block. See weighted table below
to make comparisons between layers.

'''

MIDDLE = '''
---

## Weighted

Same again, but the double block values divided by 2.4. So this is proportional to error per parameter quantized.

'''

POSTLUDE = '''
Patches from (https://huggingface.co/city96/FLUX.1-dev-gguf)
'''

def to_md(costs):
    all_casts = ['Q8_0', 'bf8', 'bnb8', 'Q5_1', 'Q4_0*', 'Q4_1', 'Q4_1*', 'Q4_K_S*', 'bnbFP4', 'bnbNF4', 'Q3_K_S*', 'Q2_K*']
    for layer in costs.values():
        for cast in layer:
            if not cast in all_casts: all_casts.append(cast)

    def format(x, divide_by=1.0):
        if not x: return ""
        return f"{x/divide_by: >7.3f}"

    with open(path("casting_cost.md"), 'w') as f:
        print(PRELUDE, file=f)

        print( "|-|" + "|".join(all_casts) + "|", file=f )
        print( "|-|" + "|".join("-:" for _ in all_casts) + "|", file=f )
        for layer in costs:
            print( f"|{layer}|" + "|".join( format(costs[layer].get(cast,None)) for cast in all_casts ) + "|", file=f)

        print(MIDDLE, file=f)

        print( "|-|" + "|".join(all_casts) + "|", file=f )
        print( "|-|" + "|".join("-:" for _ in all_casts) + "|", file=f )
        for layer in costs:
            print( f"|{layer}|" + "|".join( format(costs[layer].get(cast,None), (2.4 if int(layer)<19 else 1.0)) for cast in all_casts ) + "|", file=f)


        print(POSTLUDE, file=f)

        

to_md(from_csv())

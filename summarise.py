import os, yaml

labels = {}  # "layer{:0>2}".format(i):{} for i in range(57) }
casts = set()
dir = "output"
for file in os.listdir(dir):
    if os.path.splitext(file)[-1]==".yaml":
        x:dict
        with open(os.path.join(dir,file),'r') as f:
            x = yaml.safe_load(f)
        for layer_name, layer in x.items():
            cast, label, sensitivity = None, None, None
            for key, value in layer.items():
                if isinstance(value,str): 
                    assert cast is None or cast==value
                    cast = value
                    thelabel = layer_name + ('.txt' if 'txt' in key else ('.img' if 'img' in key else '.x'))
                    assert label is None or label==thelabel
                    label = thelabel
            casts.add(cast)
            if thelabel not in labels: labels[thelabel] = {}
            labels[thelabel][cast] = layer['sensitivity']

casts = list(casts)
label_names = [l for l in labels]
label_names.sort()
labels = { k:labels[k] for k in label_names }

with open(os.path.join(dir,'summary.csv'), 'w') as f:
    print( ",".join(["layer",]+casts), file=f)
    for layer in labels:
        sensititivities = [ str(labels[layer].get(cast,"")) for cast in casts ]
        print( ",".join([layer,]+sensititivities), file=f)

import add_paths
from modules.casting import QuantizedTensor
from modules.utils import shared, layer_iteratable_from_string
from modules.loader import load_layer_stack
from gguf import GGMLQuantizationType, GGUFWriter
import torch
from argparse import ArgumentParser

HELP_TEXT = '''Produce a mixed gguf model from a flux safetensors. 
Usage:
python convert.py --load [flux_model].safetensors --save [output model].gguf --config xx_x

Ideally use a 16 bit safetensors to load from. 

Saved files can be loaded in Comfy using the nodes at https://github.com/city96/ComfyUI-GGUF

The config is an approximate number of GB removed from the full 16 bit model.  
 9_0 might just fit on a 16GB card
10_6 is a good balance for 16GB cards,
12_0 is roughly the size of an 8 bit model,
14_1 should work for 12 GB cards
15_2 is fully quantised to Q4_1 

use
python convert.py -h
to see all current config options
'''

CONFIGURATIONS = {
    "9_0" : { 
        'casts': [
            {'layers': '0-10',             'castto': 'BF16'},
            {'layers': '11-14, 54',        'castto': 'Q8_0'},
            {'layers': '15-36, 39-53, 55', 'castto': 'Q5_1'},
            {'layers': '37-38, 56',        'castto': 'Q4_1'},
        ]
    },
    "10_6" : { 
        'casts': [
            {'layers': '0-4, 10',      'castto': 'BF16'},
            {'layers': '5-9, 11-14',   'castto': 'Q8_0'},
            {'layers': '15-35, 41-55', 'castto': 'Q5_1'},
            {'layers': '36-40, 56',    'castto': 'Q4_1'},
        ]
    },
    "12_0" : {
        'casts': [
            {'layers': '0-2',                  'castto': 'BF16'},
            {'layers': '5, 7-12',              'castto': 'Q8_0'},
            {'layers': '3-4, 6, 13-33, 42-55', 'castto': 'Q5_1'},
            {'layers': '34-41, 56',            'castto': 'Q4_1'},
        ]
    },
    "14_1" : {
        'casts': [
            {'layers': '0-25, 27-28, 44-54', 'castto': 'Q5_1'},
            {'layers': '26, 29-43, 55-56',   'castto': 'Q4_1'},
        ]
    },
    "15_2" : {
        'casts': [
            {'layers': '0-56', 'castto': 'Q4_1'},
        ]
    },
}

def convert(outfile, config):
    default_cast = 'F32'
        
    print ("Getting layer casts")
    layer_casts = [default_cast]*57
    for nod in config['casts']:
        cast = nod['castto']
        for layer_index in layer_iteratable_from_string(nod['layers']):
            layer_casts[layer_index] = cast

    layers = load_layer_stack()

    writer = GGUFWriter(outfile, "flux", use_temp_file=True)
    def write(key, tensor:torch.Tensor, cast:callable):
        cast = cast(key, tensor)
        qtype = getattr(GGMLQuantizationType, cast)
        qt = QuantizedTensor.from_unquantized_tensor(tensor, qtype)
        writer.add_tensor(key, qt._tensor.numpy(), raw_dtype=qtype)
        writer.add_array(f"comfy.gguf.orig_shape.{key}", tensor.shape)
        print(f"{key:>50} {cast:<20}")
    
    print("Casting leftovers")
    for key in shared.sd: 
        write(key, shared.sd[key], lambda a,b: default_cast )

    print("Casting layers")
    for i, layer in enumerate(layers):
        cast = layer_casts[i]
        prefix = f"double_blocks.{i}." if i<19 else f"single_blocks.{i-19}."
        sd = layer.state_dict()
        def get_cast(key:str, tensor:torch.Tensor):
            if len(tensor.shape)==1 and tensor.shape[0]<2000:
                return default_cast
            else:
                return cast
        for key in sd: write( prefix+key, sd[key], get_cast)

    print("Writing to file")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

def main():
    a = ArgumentParser()
    a.add_argument('--load', required=False, help="The model to convert. Best if it is 16bit. Safetensors format.")
    a.add_argument('--save', required=False, help="Where to save the resulting model. Saves in gguf format.")
    a.add_argument('--config', required=False, help="Configuration to use. Numbers are the approximate number of GB removed from 16 bit model.", 
                   choices=[k for k in CONFIGURATIONS])

    args = a.parse_args()

    if not (args.load and args.save and args.config):
        print(HELP_TEXT)
        return

    shared._sd = args.load
    convert(args.save, config=CONFIGURATIONS[args.config])

if __name__=='__main__': main()
    
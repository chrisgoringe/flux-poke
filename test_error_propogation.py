import add_paths
import torch, statistics, math

from modules.layer import load_single_layer
from modules.generated_dataset import TheDataset
from modules.utils import is_double, shared, log
from modules.hffs import HFFS_Cache
from modules.arguments import args
import json

class Onlies:
    IMG=0
    TXT=1
    X  =2

class Perturb(torch.nn.Module):
    def __init__(self, fac, only:Onlies=None):
        super().__init__()
        self.fac  = fac
        self.only = only
    
    def forward(self, *args):
        def should_mod(a,i):
            return (a is not None and (self.only is None or self.only==i))
        return ( (a * (1+(torch.rand_like(a)-0.5)*self.fac) if should_mod(a,i) else a) for i,a in enumerate(args)  )

LOW_VRAM = False

def calulate_error_propogation(stack, dataset:TheDataset, perturb_before:int, perturb_magnitude, tests_per_sample=1, only=None):
    
    perturb = Perturb( perturb_magnitude, only )
    losses    = []
    loss_fn   = torch.nn.MSELoss()

    for di, datum in enumerate(dataset):
        log(f"Datum {di}/{len(dataset)}")
        vec:torch.Tensor    = datum.get('vec').cuda().unsqueeze_(0)
        pe:torch.Tensor     = datum.get('pe').cuda().unsqueeze_(0)
        img0:torch.Tensor   = datum.get('img').cuda().unsqueeze_(0)
        txt0:torch.Tensor   = datum.get('txt').cuda().unsqueeze_(0)
        x_out:torch.Tensor  = datum.get('x_out').cuda().unsqueeze_(0)

        for ti in range(tests_per_sample):
            log(f"Test {ti}/{tests_per_sample} on pb {perturb_before}")
            x   = None
            img = img0.clone()
            txt = txt0.clone()
            with torch.no_grad():
                for layer_index, layer in enumerate(stack):
                    if LOW_VRAM: layer.cuda()
                    if layer_index==perturb_before: 
                        img, txt, x = perturb(img, txt, x)     

                    if is_double(layer_index): img, txt = layer( img,  txt,  vec, pe ) 
                    else:                      x        = layer( x if x is not None else torch.cat((txt,img ), dim=1),  vec, pe ) 
                    if LOW_VRAM: layer.cpu()
                loss = float(loss_fn(x, x_out))
                losses.append(loss)
                mean = statistics.mean(losses)
                stderr = (statistics.stdev(losses) if len(losses)>1 else mean)/math.sqrt(len(losses))
                print ("This loss: {:>8.4f}. Loss {:>8.4f} +/- {:>8.4f} ({:>4}/{:<4} samples)".format(
                    loss, mean, stderr, len(losses), len(dataset)*tests_per_sample))

    return statistics.mean(losses), statistics.stdev(losses)/math.sqrt(len(losses))

def main():
    stack = torch.nn.ModuleList(load_single_layer(layer_number=x, remove_from_sd=True) for x in range(57))
    dataset = TheDataset(first_layer=0, thickness=57, split='all')
    if not LOW_VRAM: stack.cuda()
    mag = 0.01
    with open('pb.txt','w') as f:
        print(f"Perturbation magnitude {mag}", file=f, flush=True)
        for pb in range(0,19):
            loss, stderr = calulate_error_propogation(stack=stack, 
                                                      dataset=dataset, 
                                                      perturb_before=pb, 
                                                      perturb_magnitude=mag, 
                                                      tests_per_sample=2,
                                                      only=Onlies.IMG)
            print("pb {:>2} (img) loss {:>8.4f} +/- {:>8.4f}".format(pb, loss, stderr), file=f, flush=True)
            loss, stderr = calulate_error_propogation(stack=stack, 
                                                      dataset=dataset, 
                                                      perturb_before=pb, 
                                                      perturb_magnitude=mag, 
                                                      tests_per_sample=2,
                                                      only=Onlies.TXT)
            print("pb {:>2} (txt) loss {:>8.4f} +/- {:>8.4f}".format(pb, loss, stderr), file=f, flush=True)

def rms():
    rmses = [{i:[] for i in range(57)}, {i:[] for i in range(57)}, {i:[] for i in range(57)}, ]

    for layer_index in range(57):
        print(layer_index)
        rms_img = rmses[0][layer_index]
        rms_txt = rmses[1][layer_index]
        rms_x = rmses[2][layer_index]
        loss = torch.nn.MSELoss()
        rms = lambda a: float(loss(a.cuda(), torch.zeros_like(a).cuda()))
        dataset = TheDataset(first_layer=layer_index, thickness=1, split='all')
        for entry in dataset:
            if is_double(layer_index):
                rms_img.append( rms(entry['img']) )
                rms_txt.append( rms(entry['txt']) )
            else:
                rms_x.append( rms(entry['x']) )

    with open('rmses.json', 'w') as f: print(json.dumps(rmses), file=f)

if __name__=='__main__': 
    shared.set_shared_filepaths(args=args)
    HFFS_Cache.set_cache_directory(args.cache_dir)
    TheDataset.set_dataset_source(dir=args.hs_dir)
    main()

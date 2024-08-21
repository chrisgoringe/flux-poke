import add_paths
import torch, statistics, math

from modules.layer import load_single_layer
from modules.generated_dataset import TheDataset
from modules.utils import is_double, shared, log
from modules.hffs import HFFS_Cache
from modules.arguments import args

class Perturb(torch.nn.Module):
    def __init__(self, fac):
        super().__init__()
        self.fac = fac
    
    def forward(self, *args):
        return ( (a * (1+torch.rand_like(a)*self.fac) if a is not None else None) for a in args  )

def calulate_error_propogation(stack, dataset, perturb_before:int, perturb_magnitude = 0.001, tests_per_sample=10):
    
    perturb = Perturb( perturb_magnitude )
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
            log(f"Test {ti}/{tests_per_sample}")
            x   = None
            img = img0.clone()
            txt = txt0.clone()
            with torch.no_grad():
                for layer_index, layer in enumerate(stack):
                    if layer_index==perturb_before: img, txt, x = perturb(img, txt, x)     

                    
                    if is_double(layer_index): img, txt = layer( img,  txt,  vec, pe ) 
                    else:                      x        = layer( x if x is not None else torch.cat((txt,img ), dim=1),  vec, pe ) 
                    
                loss = float(loss_fn(x, x_out))
                losses.append(loss)
                mean = statistics.mean(losses)
                stderr = (statistics.stdev(losses) if len(losses)>1 else mean)/math.sqrt(len(losses))
                print ("This loss: {:>8.4f}. Loss {:>8.4f} +/- {:>8.4f} ({:>4}/{:<4} samples)".format(
                    loss, mean, stderr, len(losses), len(stack)*tests_per_sample))

    return statistics.mean(losses), statistics.stdev(losses)/math.sqrt(len(losses))

def main():
    stack = torch.nn.ModuleList(load_single_layer(layer_number=x, remove_from_sd=True) for x in range(57))
    stack.cuda()
    dataset = TheDataset(first_layer=0, thickness=57, split='eval', train_frac=0.0)
    with open('pb.txt','w') as f:
        for pb in range(57):
            loss, stderr = calulate_error_propogation(stack, dataset, perturb_before=pb, tests_per_sample=12)
            print("pb {:>2} loss {:>8.4f} +/- {:>8.4f}".format(pb, loss, stderr), file=f, flush=True)

if __name__=='__main__': 
    shared.set_shared_filepaths(args=args)
    HFFS_Cache.set_cache_directory(args.cache_dir)
    TheDataset.set_dataset_source(dir=args.hs_dir)
    main()

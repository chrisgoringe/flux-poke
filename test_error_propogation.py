import add_paths
import torch, statistics, math

from modules.layer import load_single_layer
from modules.generated_dataset import TheDataset
from modules.utils import is_double, shared
from modules.hffs import HFFS_Cache
from tqdm import tqdm

class Perturb(torch.nn.Module):
    def __init__(self, fac):
        super().__init__()
        self.fac = fac
    
    def forward(self, *args):
        return ( (a * (1+torch.rand_like(a)*self.fac) if a is not None else None) for a in args  )
        
#LAYERS_ON_CUDA = 32

def two_paths(stack, dataset, perturb_before:int, perturb_magnitude = 0.001):
    
    perturb = Perturb( perturb_magnitude )
    losses    = []
    loss_fn   = torch.nn.MSELoss()

    for datum in dataset:
        vec:torch.Tensor   = datum.get('vec').cuda().unsqueeze_(0)
        pe:torch.Tensor    = datum.get('pe').cuda().unsqueeze_(0)
        img:torch.Tensor   = datum.get('img').cuda().unsqueeze_(0)
        txt:torch.Tensor   = datum.get('txt').cuda().unsqueeze_(0)
        x:torch.Tensor     = None
        x_out:torch.Tensor = datum.get('x_out').unsqueeze_(0)

        with torch.no_grad():
            for layer_index, layer in enumerate(stack):
                print(f"\r{layer_index}", end='')
                if layer_index==perturb_before: img, txt, x = perturb(img, txt, x)     

                layer.cuda()
                if is_double(layer_index): img, txt = layer( img,  txt,  vec, pe ) 
                else:                      x        = layer( x if x is not None else torch.cat((txt,img ), dim=1),  vec, pe ) 
                #if layer_index>LAYERS_ON_CUDA: layer.cpu()
                
            loss = float(loss_fn(x, x_out.cuda()))
            losses.append(loss)
            mean = statistics.mean(losses)
            stderr = (statistics.stdev(losses) if len(losses)>1 else mean)/math.sqrt(len(losses))
            print ("\nThis loss: {:>8.4f}. Loss {:>8.4f} +/- {:>8.4f} ({:>4} samples)".format(loss, mean, stderr, len(losses)))

    return statistics.mean(losses), statistics.stdev(losses)/math.sqrt(len(losses))

def main():
    stack = [load_single_layer(layer_number=x) for x in range(57)]
    dataset = TheDataset(first_layer=0, thickness=57, split='eval', train_frac=0.0)
    with open('pb.txt','w') as f:
        for pb in range(57):
            loss, stderr = two_paths(stack, dataset, perturb_before=pb)
            print("pb {:>2} loss {:>8.4f} +/- {:>8.4f}".format(pb, loss, stderr), file=f)

if __name__=='__main__': 
    class FakeArgs:
        model = 'D:/models/unet/flux1-dev.sft'
        internals = ''

    shared.set_shared_filepaths(args=FakeArgs())
    HFFS_Cache.set_cache_directory("e:/.hfc")
    TheDataset.set_dataset_source(dir="ChrisGoringe/fi")
    main()

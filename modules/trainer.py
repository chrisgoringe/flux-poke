import transformers
import torch

class TheTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()
        self.can_return_loss = True

    def check_inf(self,*args):
        for a in args:
            if torch.isinf(a).any():
                print("INF")

    def compute_loss(self, model:torch.nn.Sequential, inputs:dict[str,torch.Tensor], return_outputs=False):
        vec, pe = inputs['vec'], inputs['pe']

        if 'img' in inputs and 'img_out' in inputs: # DoubleStreamBlock
            img, txt, img_out, txt_out = inputs['img'], inputs['txt'], inputs['img_out'],inputs['txt_out']
            for layer in model: img, txt = layer( img, txt, vec, pe ) 
            self.check_inf(img, txt)
            loss = self.loss_fn(torch.cat((img,txt),dim=1), torch.cat((img_out, txt_out), dim=1))
            return (loss, (img, txt)) if return_outputs else loss
        
        else:
            x_out = inputs['x_out']
            x     = inputs['x']

            for layer in model: x = layer( x, vec, pe ) 
            self.check_inf(x)
            loss = self.loss_fn(x, x_out)
            return (loss, x) if return_outputs else loss
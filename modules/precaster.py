import threading, queue, torch, math
from .casting import CastLinear, DequantingLinear
from .utils import SingletonAddin

class Precaster(SingletonAddin):
    def __init__(self):
        self.queue = queue.SimpleQueue()
        self.enabled = True
        threading.Thread(group=None, target=self.runner, daemon=True).start()

    def runner(self):
        while True:
            target, dtype, device, message = self.queue.get()
            if message: print(f"[{message}] retrieved from queue")
            count = 0
            for m in target.modules():
                if isinstance(m, CastLinear):
                    if isinstance(m.linear, DequantingLinear):
                        weight, bias = m.linear.get_weight_and_bias(dtype, device)
                        count += math.prod(weight.shape)
                        if bias is not None: count += math.prod(bias.shape)
            if message: print(f"[{message}] complete - {count} parameters precast")


    def precast(self, target, dtype=torch.bfloat16, device="cuda", message=None):
        if not self.enabled: return
        if message: print(f"[{message}] added to queue")
        self.queue.put((target, dtype, device, message))

precaster = Precaster.instance()
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
            target, dtype, device = self.queue.get()
            for m in target.modules():
                if isinstance(m, CastLinear):
                    if isinstance(m.linear, DequantingLinear):
                        m.linear.get_weight_and_bias(dtype, device)

    def precast(self, target, dtype=torch.bfloat16, device="cuda"):
        if not self.enabled: return
        self.queue.put((target, dtype, device))

precaster = Precaster.instance()
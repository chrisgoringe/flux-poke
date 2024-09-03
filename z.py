from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader, GGMLQuantizationType
import math


reader = GGUFReader("d:/models/unet/gguf/flux1-dev-Q2_K.gguf")
expect = GGMLQuantizationType.Q4_1
tensor:ReaderTensor


for tensor in reader.tensors:
    print(f"{tensor.name}")         




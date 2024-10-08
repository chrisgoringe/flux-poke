from gguf.gguf_reader import ReaderTensor
from gguf import GGUFReader, GGMLQuantizationType
import math


reader = GGUFReader("d:/models/unet/gguf/flux1-dev-Q6_K.gguf")
expect = GGMLQuantizationType.Q4_1
tensor:ReaderTensor


for tensor in reader.tensors:

    n = math.prod(tensor.shape)
    m = tensor.n_bytes
    print(f"{tensor.name} {tensor.tensor_type.name} {8*m/n}")         




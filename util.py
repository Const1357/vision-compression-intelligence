import torch
import numpy as np
import random

SEED = 12345
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# seeding
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

MODEL_TYPES = ['VQ-GAN', 'RQ-Transformer', 'LlamaGen', 'VAR']

MODEL_SIZES = {
    'VQ-GAN': [
        '1.4B'
    ],
    'RQ-Transformer': [
        '481M',
        '821M',
        '1.4B'
    ],
    'LlamaGen': [
        '111M_256 (B)',
        '111M_384 (B)',
        '343M_256 (L)',
        '343M_384 (L)',
        '775M_384 (XL)',
        '1.4B_384 (XXL)',
    ],

    'VAR': [
        '310M',
        '600M',
        '1B',
        '2B',
    ],
}


# diagnostics
def mib(x): return x / 1024**2
def gib(x): return x / 1024**3

def report_cuda(tag=""):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    m = torch.cuda.max_memory_allocated()
    print(f"[{tag}] allocated={gib(a):.2f} GiB | reserved={gib(r):.2f} GiB | peak_alloc={gib(m):.2f} GiB")





import Model

from interfaces.VQGAN import get_model_VQGAN
from interfaces.RQTransformer import get_model_RQTransformer
from interfaces.LlamaGen import get_model_LlamaGen
from interfaces.VAR import get_model_VAR

def get_model(model_type, size) -> Model:

    if model_type == 'VAR':
        return get_model_VAR(size)
    elif model_type == 'LlamaGen':
        return get_model_LlamaGen(size)
    elif model_type == 'VQ-GAN':
        return get_model_VQGAN(size)
    elif model_type == 'RQ-Transformer':
        return get_model_RQTransformer(size)
    else:
        raise ValueError(f"Model {model_type} not recognized.")
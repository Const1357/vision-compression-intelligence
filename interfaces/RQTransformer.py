import sys
from Model import Model
from util import *

from models_src.RQTransformer.rqvae.utils.config import load_config

from models_src.RQTransformer.rqvae.models.rqvae import RQVAE
from models_src.RQTransformer.rqvae.models.rqtransformer import RQTransformer


RFID = 4.73 # tokenizer rFID

# Config Paths
SIZE_TO_TOKENIZER_CONFIG_PTH = {
    '481M' : 'checkpoints/RQTransformer/imagenet_480M/stage1/config.yaml',
    '821M' : 'checkpoints/RQTransformer/imagenet_821M/stage1/config.yaml',
    '1.4B' : 'checkpoints/RQTransformer/imagenet_1.4B/stage1/config.yaml',
}
SIZE_TO_TRANSFORMER_CONFIG_PTH = {
    '481M' : 'checkpoints/RQTransformer/imagenet_480M/stage2/config.yaml',
    '821M' : 'checkpoints/RQTransformer/imagenet_821M/stage2/config.yaml',
    '1.4B' : 'checkpoints/RQTransformer/imagenet_1.4B/stage2/config.yaml',
}

# Checkpoint Paths
SIZE_TO_CHECKPOINT_PTH = {
    '481M' : 'checkpoints/RQTransformer/imagenet_480M/stage2/model.pt',
    '821M' : 'checkpoints/RQTransformer/imagenet_821M/stage2/model.pt',
    '1.4B' : 'checkpoints/RQTransformer/imagenet_1.4B/stage2/model.pt',
}
SIZE_TO_TOKENIZER_CHECKPOINT_PTH = {
    '481M' : 'checkpoints/RQTransformer/imagenet_480M/stage1/model.pt',
    '821M' : 'checkpoints/RQTransformer/imagenet_821M/stage1/model.pt',
    '1.4B' : 'checkpoints/RQTransformer/imagenet_1.4B/stage1/model.pt',
}

# FID
SIZE_TO_FID = {
    '481M': 15.72,
    '821M': 13.11,
    '1.4B': 11.56,
}

class RQTransformerModel(Model):
    def __init__(self, model, tokenizer, model_type="RQ-Transformer", rfid=0.0, fid=0.0, channels=3, img_size=256):
        super(RQTransformerModel, self).__init__(model, tokenizer, model_type, rfid, fid, channels, img_size)

        self.model: RQTransformer
        self.tokenizer: RQVAE

    def get_nll(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        # TODO: Implement NLL computation for RQ-Transformer

        x = x.to(DEVICE)
        B = x.shape[0]

        with torch.no_grad():
            B = x.shape[0]
            codes = self.tokenizer.get_codes(x)

            if C is None:
                C = torch.zeros(B, dtype=torch.long, device=self.device)
            elif isinstance(C, int):
                C = torch.full((B,), C, dtype=torch.long, device=self.device)

            # Forward pass: (B, H, W, D, VocabSize)
            logits = self.model(codes, model_aux=self.tokenizer, cond=C)

            # Flatten for cross entropy calculation
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = codes.reshape(-1)

            loss_per_token = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')
            return loss_per_token.view(B, -1).sum(dim=1)
  

def get_model_RQTransformer(size) -> RQTransformerModel:
    
    transformer_checkpoint_pth = SIZE_TO_CHECKPOINT_PTH.get(size, None)
    tokenizer_checkpoint_pth = SIZE_TO_TOKENIZER_CHECKPOINT_PTH.get(size, None)
    fid = SIZE_TO_FID.get(size, None)
    rfid = RFID

    model_config_pth = SIZE_TO_TRANSFORMER_CONFIG_PTH.get(size, None)
    tokenizer_config_pth = SIZE_TO_TOKENIZER_CONFIG_PTH.get(size, None)

    if transformer_checkpoint_pth is None or tokenizer_checkpoint_pth is None or model_config_pth is None or tokenizer_config_pth is None:
        raise ValueError(f"Model size {size} not recognized.", file=sys.stderr)
    
    tokenizer: RQVAE
    transformer: RQTransformer

    # build vqvae tokenizer and vqtransformer model
    tokenizer_config = load_config(tokenizer_config_pth)
    model_config = load_config(model_config_pth)

    tokenizer = RQVAE(**tokenizer_config.arch.hparams, ddconfig=tokenizer_config.arch.ddconfig)
    transformer = RQTransformer(model_config.arch)

    transformer.to(DEVICE)
    tokenizer.to(DEVICE)

    # load checkpoints
    transformer.load_state_dict(torch.load(transformer_checkpoint_pth, map_location=DEVICE)['state_dict'])
    tokenizer.load_state_dict(torch.load(tokenizer_checkpoint_pth, map_location=DEVICE)['state_dict'])

    for p in tokenizer.parameters(): p.requires_grad_(False)
    for p in transformer.parameters(): p.requires_grad_(False)
    tokenizer.eval(); transformer.eval()

    model = RQTransformerModel(transformer, tokenizer, rfid=rfid, fid=fid, img_size=256)
    return model

    
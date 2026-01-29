import sys

sys.path.append('models_src/VQGAN.taming')
sys.path.append('models_src/VQGAN')

from Model import Model

from util import *
from omegaconf import OmegaConf

from models_src.VQGAN.taming.models.cond_transformer import Net2NetTransformer  # circular import ???


CONFIG_PTH = 'checkpoints/VQGAN/configs.yaml'
CHECKPOINTS_PTH = 'checkpoints/VQGAN/last.ckpt'
FID = 15.78
RFID = 4.98


class VQGAN_Model(Model):
    def __init__(self, model, tokenizer, model_type="VQGAN", rfid=0.0, fid=0.0, channels=3, img_size=256):
        super(VQGAN_Model, self).__init__(model, tokenizer, model_type, rfid, fid, channels, img_size)


    def get_nll(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        """
        Calculates the negative log likelihood per image
        """

        x = x.to(DEVICE)
        B = x.shape[0]


        # Ensure inputs are on the GPU
        x = x.to(DEVICE)

        #Standardize 'C' condition to be a tensor of shape (Batch_Size,)
        if isinstance(C,int):
            C= torch.full((x.shape[0]), C, DEVICE=DEVICE, dtype=torch.long)
        elif isinstance(C, torch.LongTensor):
            C = C.to(DEVICE)
            if C.ndim == 0:
                C = C.expand(x.shape[0])

        # we dont need gradient for eval
        with torch.no_grad():
            # Tokenization
            # turn images into codebook indices
            # encode_to_z returns (quantized_embeds, indices).
            _, z_indices = self.model.encode_to_z(x)

            # Turn class labels into condition indices
            _, c_indices = self.model.encode_to_c(C)

            # Sequence Construction
            # [condition tokens, image tokens]
            # example shape: [batch_size, 1+256]
            cz_indices = torch.cat((c_indices, z_indices), dim=1)

            # Teacher forcing (forward pass)
            logits, _ = self.model.transformer(cz_indices[:, :-1])

            # Align output
            target = z_indices

            # calculate loss
            #flatten to (batch*sequence, Vocab) for cross entropy
            loss_per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction='none'
            )

            # reduce back to batch, sequence ->[B, 256]
            loss_per_token = loss_per_token.view(x.shape[0], -1)

            # sum over sequence to get per-image NLL
            return loss_per_token.sum(dim=1)
    

def get_model_VQGAN(size: str) -> VQGAN_Model:

    fid = FID
    rfid = RFID

    config = OmegaConf.load(CONFIG_PTH)
    model = Net2NetTransformer(**config.model.params).to(DEVICE)

    checkpoint = torch.load(CHECKPOINTS_PTH, map_location=DEVICE, weights_only=False)['state_dict']
    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    tokenizer = model.first_stage_model
    transformer = model

    for p in tokenizer.parameters(): p.requires_grad_(False)
    for p in transformer.parameters(): p.requires_grad_(False)
    tokenizer.eval(); transformer.eval()


    model = VQGAN_Model(transformer, tokenizer, rfid=rfid, fid=fid, img_size=256)
    return model


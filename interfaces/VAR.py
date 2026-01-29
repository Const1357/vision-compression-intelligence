import sys
from Model import Model
from util import *

from models_src.VAR.models import build_vae_var

SIZE_TO_PTH = {
    '310M': 'checkpoints/VAR/var_d16.pth',
    '600M': 'checkpoints/VAR/var_d20.pth',
    '1B': 'checkpoints/VAR/var_d24.pth',
    '2B': 'checkpoints/VAR/var_d30.pth',
}

SIZE_TO_DEPTH = {
    '310M': 16,
    '600M': 20,
    '1B': 24,
    '2B': 30,
}

# for reference check the VAR paper
RFID = 1.78 # tokenizer rFID
SIZE_TO_FID = {
    '310M': 3.3,
    '600M': 2.57,
    '1B': 2.09,
    '2B': 1.92,
}

TOKENIZER_checkpoint_pth = 'checkpoints/VAR/vae_ch160v4096z32.pth'

from models_src.VAR.models import VQVAE, VAR
from Model import Model


class VAR_Model(Model):
    def __init__(self, model, tokenizer, model_type="VAR", rfid=0.0, fid=0.0, channels=3, img_size=256):
        super(VAR_Model, self).__init__(model, tokenizer, model_type, rfid, fid, channels, img_size)

        self.model: VAR
        self.tokenizer: VQVAE

        # Alias for NLL
        self.vae = self.tokenizer
        self.var_model = self.model

    def forward(self, x):
        # find a way to get the training loss and compare to nll computed in get_nll.
        self.model.train()
        return self.model(x)
    
    def get_nll(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        """Calculates NLL via Next-Scale Prediction teacher forcing."""
        x = x.to(DEVICE)
        if isinstance(C, int):
            label_B = torch.full((x.shape[0],), C, device=DEVICE, dtype=torch.long)
        else:
            label_B = C.to(DEVICE).view(-1)

        with torch.no_grad():
            # 1. Get Ground Truth Indices
            gt_idx_Bl_list = self.vae.img_to_idxBl(x)
            targets = torch.cat(gt_idx_Bl_list, dim=1)  # [B, Total_Length]

            # 2. Construct teacher forcing input
            B = x.shape[0]
            Cvae = self.vae.Cvae
            patch_nums = self.var_model.patch_nums
            quantizer = self.vae.quantize

            f_hat = torch.zeros(B, Cvae, patch_nums[-1], patch_nums[-1], device=DEVICE)
            input_vectors_list = []

            for k in range(len(patch_nums) - 1): # up to second to last scale
                idx_k = gt_idx_Bl_list[k]  # [B, Lk]
                h_k = quantizer.embedding(idx_k)  # [B, Lk, Cvae]

                # reshape to (Batch, Channels, Height, Width)
                pn = patch_nums[k]
                h_k = h_k.transpose(1, 2).reshape(B, Cvae, pn, pn)

                # Update f_hat and get conditional map for next scale
                f_hat, next_token_map = quantizer.get_next_autoregressive_input(k, len(patch_nums), f_hat, h_k)
                input_vectors_list.append(next_token_map.view(B, Cvae, -1).transpose(1, 2))

            x_BLCv_wo_first_l = torch.cat(input_vectors_list, dim=1)

            # 3. Forward Pass & NLL Calculation
            logits = self.var_model(label_B, x_BLCv_wo_first_l)
            loss_per_token = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none')

            loss_per_image = loss_per_token.view(B, -1).sum(dim=1)

            return loss_per_image


def get_model_VAR(size) -> VAR_Model:

    checkpoint_pth = SIZE_TO_PTH.get(size, None)
    depth = SIZE_TO_DEPTH.get(size, None)
    fid = SIZE_TO_FID.get(size, None)
    rfid = RFID


    if checkpoint_pth is None:
        print(f"Model size {size} not found for VAR. Valid sizes are: {list(SIZE_TO_PTH.keys())}", file=sys.stderr)
        return None


    tokenizer: VQVAE
    var_model: VAR
    
    # CODE FROM https://github.com/FoundationVision/VAR/blob/main/demo_sample.ipynb

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer, var_model = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=depth, shared_aln=False,    # num_classes = 1000 for ImageNet
    )

    # load checkpoints
    var_model.load_state_dict(torch.load(checkpoint_pth, map_location=DEVICE))
    tokenizer.load_state_dict(torch.load(TOKENIZER_checkpoint_pth, map_location=DEVICE))
    
    # sanity
    tokenizer.to(DEVICE)
    var_model.to(DEVICE)

    for p in tokenizer.parameters(): p.requires_grad_(False)
    for p in var_model.parameters(): p.requires_grad_(False)
    tokenizer.eval(), var_model.eval()


    model = VAR_Model(var_model, tokenizer, rfid=rfid, fid=fid, img_size=256)
    return model
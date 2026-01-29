import sys
from Model import Model
from util import *

from models_src.LlamaGen.tokenizer.tokenizer_image.vq_model import VQ_models, VQModel
from models_src.LlamaGen.autoregressive.models.gpt import GPT_models, Transformer

SIZE_TO_PTH = {
    '111M_256 (B)': ('checkpoints/LlamaGen/c2i_B_256.pt', 256),
    '111M_384 (B)': ('checkpoints/LlamaGen/c2i_B_384.pt', 384),
    '343M_256 (L)': ('checkpoints/LlamaGen/c2i_L_256.pt', 256),
    '343M_384 (L)': ('checkpoints/LlamaGen/c2i_L_384.pt', 384),
    '775M_384 (XL)': ('checkpoints/LlamaGen/c2i_XL_384.pt', 384),
    '1.4B_384 (XXL)': ('checkpoints/LlamaGen/c2i_XXL_384.pt', 384),
}

SIZE_TO_GPT_KEY = {
    '111M_256 (B)': 'GPT-B',
    '111M_384 (B)': 'GPT-B',
    '343M_256 (L)': 'GPT-L',
    '343M_384 (L)': 'GPT-L',
    '775M_384 (XL)': 'GPT-XL',
    '1.4B_384 (XXL)': 'GPT-XXL',
}


# for reference check the LlamaGen paper
SIZE_TO_RFID = {
    '111M_256 (B)': 2.19,
    '111M_384 (B)': 0.94,
    '343M_256 (L)': 2.19,
    '343M_384 (L)': 0.94,
    '775M_384 (XL)': 0.94,
    '1.4B_384 (XXL)': 0.94,
}

SIZE_TO_TOKEN_SIZE = {
    '111M_256 (B)': 16,
    '111M_384 (B)': 24,
    '343M_256 (L)': 16,
    '343M_384 (L)': 24,
    '775M_384 (XL)': 24,
    '1.4B_384 (XXL)': 24,
}

SIZE_TO_FID = {
    '111M_256 (B)': 8.69,
    '111M_384 (B)': 12.89,
    '343M_256 (L)': 4.21,
    '343M_384 (L)': 5.01,
    '775M_384 (XL)': 3.42,
    '1.4B_384 (XXL)': 2.89,
}

TOKENIZER_checkpoint_pth = 'checkpoints/LlamaGen/vq_ds16_c2i.pt'
# single tokenizer checkpoint.
# VQ-16 tokenizer used for all model sizes.
# token size (16x16 or 24x24) changes based on image size (according to documentation: 16x16 for 256x256 images, 24x24 for 384x384 images).


class LlamaGen_Model(Model):
    def __init__(self, model, tokenizer, model_type="LlamaGen", rfid=0.0, fid=0.0, channels=3, img_size=256):
        super(LlamaGen_Model, self).__init__(model, tokenizer, model_type, rfid, fid, channels, img_size)

        self.model: Transformer
        self.tokenizer: VQModel
    
    def get_nll(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        x = x.to(DEVICE)
        if isinstance(C, int):
            labels = torch.full((x.shape[0],), C, device=DEVICE, dtype=torch.long)
        else:
            labels = C.to(DEVICE).view(-1)

        with torch.no_grad():
            _, _, info = self.tokenizer.encode(x)
            gt_indices = info[2].view(x.shape[0], -1)
            logits, _ = self.model(idx=gt_indices, cond_idx=labels)

            logits = logits[:, :-1, :].contiguous() # Keep first 576 logits [B, 576, Vocab]

            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), gt_indices.reshape(-1), reduction='none')
            return loss.view(x.shape[0], -1).sum(dim=1)





def get_model_LlamaGen(size) -> LlamaGen_Model:

    checkpoint_pth, img_size = SIZE_TO_PTH.get(size, (None, None))


    fid = SIZE_TO_FID.get(size, None)
    rfid = SIZE_TO_RFID.get(size, None)


    if checkpoint_pth is None:
        print(f"Model size {size} not found for LlamaGen. Valid sizes are: {list(SIZE_TO_PTH.keys())}", file=sys.stderr)
        return None


    tokenizer: VQModel
    gpt: Transformer
    

    # build VQ, GPT

    # (defaults for c2i - class to image, conditional generation.
    tokenizer = VQ_models["VQ-16"]().to(DEVICE)    
    gpt = GPT_models[SIZE_TO_GPT_KEY[size]](
        num_classes=1000,   # ImageNet classes
        model_type="c2i",
        block_size=SIZE_TO_TOKEN_SIZE[size] ** 2,   # token size: 16x16=256 tokens (for 256x256 img size), 24x24=576 (for 384x384 img size) as per documentation
    ).to(DEVICE)


    # sanity
    tokenizer.to(DEVICE)
    gpt.to(DEVICE)

    # load checkpoints

    tokenizer_chkpt = torch.load(TOKENIZER_checkpoint_pth, map_location=DEVICE)['model']
    gpt_chkpt = torch.load(checkpoint_pth, map_location=DEVICE)

    if len(gpt_chkpt.keys()) == 1 and 'model' in gpt_chkpt: # 1.4B_384 is a dict for model directly
        gpt_chkpt = gpt_chkpt['model']

    if "freqs_cis" in gpt_chkpt:
        # We verify it exists and then remove it because our Transformer class
        # in gpt.py already precomputes it in __init__
        # print("Found 'freqs_cis' in checkpoint. Removing it to allow strict loading of actual weights.")
        gpt_chkpt.pop("freqs_cis")
        
    tokenizer.load_state_dict(tokenizer_chkpt)
    gpt.load_state_dict(gpt_chkpt)
    tokenizer.eval(), gpt.eval()
    for p in tokenizer.parameters(): p.requires_grad_(False)
    for p in gpt.parameters(): p.requires_grad_(False)

    model = LlamaGen_Model(gpt, tokenizer, rfid=rfid, fid=fid, img_size=img_size)
    return model
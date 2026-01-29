from util import *
from abc import abstractmethod


# base interface (abstract) class to be extended
class Model(torch.nn.Module):
    def __init__(self, model, tokenizer, model_type, rfid=0.0, fid=0.0, channels=3, img_size=256):

        super(Model, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        self.rfid = rfid  # reconstruction FID (tokenizer)
        self.fid = fid

        self.channels = channels
        self.img_size = img_size

        self.pixels = self.img_size * self.img_size # assumming square images
        self.pixels_channels = self.pixels * self.channels

    
    @abstractmethod
    def get_nll(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        """Returns the class-conditional negative log-likelihood of x given class C.\\
        Output is per-sample NLL (in nats-using ln): shape (B,). No batch-wise reduction => per-image NLLs.\\
        Computes: -ln p(x|C).
        """
        raise NotImplementedError
    
    def get_log2_p(self, x: torch.Tensor, C: int | torch.LongTensor) -> torch.Tensor:
        """Returns the class-conditional log likelihood of x given class C in base 2.\\
        Output is per-sample log2 P(X|C). No batch-wise reduction => per-image log2 P(X|C).\\
        Computes: log2 p(x|C)."""
        nll = self.get_nll(x, C)
        return -nll / np.log(2)

    def get_bpp(self, x: torch.Tensor, C: int | torch.LongTensor) -> float:
        """Returns the bits-per-pixel (BPP) of x under the model.\\
        Computed as log2 p(x|C) / (num_pixels * num_channels).\\
        Output is per-sample BPP: shape (B,). No batch-wise reduction."""
        return -self.get_log2_p(x, C) / self.pixels_channels



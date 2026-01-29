# validator.py

# Imports
import gc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model import Model

# from util import report_cuda  # diagnostics

class ModelValidator:
    def __init__(self, model: Model, val_loader: DataLoader):
        # Intialize
        self.model = model
        self.val_loader = val_loader
        
        # Get Device
        self.device = next(model.parameters()).device

    def run_compression_eval(self):
        """
        Computation of Compression Metric (BPP)
        """
        print(f"--- Starting Compression Evaluation (BPP) ---")
        total_bpp = 0.0
        num_batches = 0
        bpp_history = []
        batch_size = 0
        
        with torch.inference_mode():
            for batch_idx, (images, labels) in enumerate(tqdm(self.val_loader, desc="Calculating BPP")):


                # Initializations
                images = images.to(self.device)
                labels = labels.to(self.device)


                # diagnostics
                # if batch_idx == 0:
                #     report_cuda(tag="Batch loaded to GPU")

                # Initialize batch size
                if batch_idx == 0:
                    batch_size = images.size(0) 
                
                # Compute BPP
                batch_bpp = self.model.get_bpp(images, labels)

                # Updates
                total_bpp += batch_bpp.sum().item()
                num_batches += images.size(0)
                batch_size_curr = images.size(0)

                # Note History of BPPs
                bpp_history.append(batch_bpp.sum().item() / batch_size_curr)


        if num_batches > 0:
            avg_bpp = total_bpp / num_batches
        else:
            avg_bpp = 0.0

        return {'model_bpp': avg_bpp,
                'model_fid': self.model.fid,
                'tokenizer_rfid': self.model.rfid,
                'bpp_history': bpp_history,
                'total_samples': num_batches,
                'batch_size': batch_size}
    
    def validate(self):
        # Run compression evaluation
        bpp = self.run_compression_eval()
        
        return bpp
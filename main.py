# Run as python main.py [mode] [model_type] [model_size]
# Examples:
#   python main.py                          -> runs all models on validation
#   python main.py all                      -> runs all models on validation
#   python main.py single VQ-GAN 1.4B       -> runs VQ-GAN on 1.4B parameters

import torch
from torchvision import datasets, transforms
import datetime
import json
from torch.utils.data import DataLoader
import os
import sys
from get_model import get_model
from util import DEVICE, MODEL_TYPES, MODEL_SIZES, report_cuda
from validator import ModelValidator

import gc   # for garbage collection, freeing up cached memory from torch after model loading to have space for batch images

def get_val_loader(imagenet_root, img_size, batch_size=32, num_workers=8):
    """
    Creates a DataLoader with transforms appropriate for the model's image size.
    All models in this project (VQGAN, RQTransformer, VAR, LlamaGen) expect [-1, 1] normalization.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Check if path exists to dataset exists
    if os.path.exists(imagenet_root):
        val_dataset = datasets.ImageNet(root=imagenet_root, split='val', transform=transform)
    else:
        raise FileNotFoundError(f"ImageNet root path '{imagenet_root}' does not exist.")
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader

def save_results(model_type, model_size, metrics, img_size):
    """
    Saves metrics and diagnostics to results/<model_type>/<model_type>_<model_size>/
    """
    # Create all folders
    base_dir = "results"
    type_dir = os.path.join(base_dir, model_type)
    specific_dir = os.path.join(type_dir, f"{model_type}_{model_size}")
    os.makedirs(specific_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # File 1: JSON file with metrics
    json_path = os.path.join(specific_dir, f"metrics_{model_type}_{model_size}.json")
    save_data = {
        "model_type": model_type,
        "model_size": model_size,
        "img_size": img_size,
        "timestamp": timestamp,
        "metrics": metrics
    }
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=4)
        
    # File 2: Info of model
    txt_path = os.path.join(specific_dir, f"diagnostics_{model_type}_{model_size}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Diagnostics for {model_type} ({model_size})\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Image Size: {img_size}x{img_size}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Final BPP: {metrics.get('model_bpp', 'N/A'):.12f}\n")
        f.write(f"Total Samples Processed: {metrics.get('total_samples', 'N/A')}\n")
        f.write("-" * 30 + "\n")
        f.write("Run completed successfully.\n")

    print(f"Results saved to {specific_dir}")

# Helper function to run validation
def validate_model(model_type, model_size, imagenet_root, batch_size=32):

    # report_cuda(tag="Initial State")      # diagnostics
    # clearing GPU cache to reduce memory fragmentation
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    # report_cuda(tag="After empty_cache")  # diagnostics

    print(f"\nLoading Model: {model_type} ({model_size})...")

    # load model
    model = get_model(model_type, model_size)
    model.to(DEVICE)
    model.eval()
    
    # retrieve image size from model instance (handles LlamaGen's 384 vs others 256)
    img_size = getattr(model, 'img_size', 256)
    print(f"Model configured for image size: {img_size}x{img_size}")

    # report_cuda(tag="after model.to(cuda)")  # diagnostics

    # Free cache, not model => free up reserved memory from model loading => batches+model can both fit in GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # report_cuda(tag="after empty_cache (model still resident)")  # diagnostics
    
    # Initialize DataLoader with correct size
    val_loader = get_val_loader(imagenet_root, img_size, batch_size=batch_size, num_workers=8)
    
    # Initialize Validator
    validator = ModelValidator(model, val_loader)

    
    # Run the validation loop
    metrics = validator.validate()

    # Save results
    save_results(model_type, model_size, metrics, img_size)
    
    print(f"\n--- Validation Results for {model_type} {model_size} ---")
    print(f"Average BPP among all samples: {metrics['model_bpp']:.12f} || Total Samples: {metrics['total_samples']}")

# Main Function
def main():
    print(f"--- Running Validation on {DEVICE} ---")

    # Arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    model_type_arg = sys.argv[2] if len(sys.argv) > 2 else None
    model_size_arg = sys.argv[3] if len(sys.argv) > 3 else None
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    

    imagenet_root = 'dataset/imagenet-val'
    
    # Choice 1: Run All Models
    if mode == 'all':
        for model_type in MODEL_TYPES:
            for model_size in MODEL_SIZES[model_type]:
                validate_model(model_type, model_size, imagenet_root, batch_size=batch_size)
                
    else:
        if model_type_arg is None or model_size_arg is None:
            print("Error: Arguments where not provided correctly for model validation.")
            return
        
        if model_type_arg not in MODEL_TYPES or model_size_arg not in MODEL_SIZES[model_type_arg]:
            print(f"Error: Invalid model name or size. Available models: {MODEL_TYPES} and sizes: {MODEL_SIZES}")
            return
        
        validate_model(model_type_arg, model_size_arg, imagenet_root, batch_size=batch_size)

if __name__ == "__main__":

    main()
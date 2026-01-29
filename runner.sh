#!/bin/bash

# Define list of models and sizes as "ModelName|ModelSize" pairs
declare -a experiments=(
    "RQ-Transformer|481M|32"
    "RQ-Transformer|821M|32"
    "RQ-Transformer|1.4B|16"
    "LlamaGen|111M_256 (B)|32"
    "LlamaGen|111M_384 (B)|16"
    "LlamaGen|343M_256 (L)|32"
    "LlamaGen|343M_384 (L)|16"
    "LlamaGen|775M_384 (XL)|16"
    "LlamaGen|1.4B_384 (XXL)|16"
    "VAR|310M|32"
    "VAR|600M|32"
    "VAR|1B|32"
    "VAR|2B|20"
    "VQ-GAN|1.4B|16"
)

# Loop through the array
for item in "${experiments[@]}"; do
    # Split the two components
    model_name="${item%%|*}"
    model_size="${item#*|}"
    model_size="${model_size%%|*}"
    batch_size="${item##*|}"

    echo "------------------------------------------"
    echo "Processing: $model_name ($model_size) with batch size $batch_size"
    
    # Run the process for the model
    python main.py single "$model_name" "$model_size" "$batch_size"
done
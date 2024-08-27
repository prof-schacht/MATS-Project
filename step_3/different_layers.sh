#!/bin/bash

# Array of layer numbers to process
layers=(0 5 10 15 20 25 30)

common_args="--wandb_project 'sae_phi3_multi_layer' --context_size 512 --expansion_factor 32 --l1_coefficient 15 --checkpoint_path 'mats_checkpoints'"


# Loop through each layer
for layer in "${layers[@]}"; do
    echo "Starting training for layer $layer"
    
    # Construct the hook_name
    hook_name="blocks.${layer}.hook_resid_post"

    echo "Starting trining for hook_name $hook_name"
    
    # Run the Python script with the current layer
    python sae_multi_sweep.py $common_args --hook_name "$hook_name" --hook_layer $layer
    
    echo "Finished training for layer $layer"
    echo "-----------------------------------"
done

echo "All layers processed successfully"
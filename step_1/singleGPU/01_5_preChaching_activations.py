from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig
import torch
from transformers import AutoConfig
import os
import shutil
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



model_name = "microsoft/Phi-3-mini-4k-instruct"
dataset_path = "mlfoundations/dclm-baseline-1.0"

model_config = AutoConfig.from_pretrained(model_name)
print(f"Number of Layers: { model_config.num_hidden_layers}" )

last_layer = model_config.num_hidden_layers - 1

total_training_steps = 100_000
batch_size = 2048
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")



new_cached_activations_path = (
    f"./cached_activations/{model_name}/{dataset_path.split('/')[-1]}/{total_training_steps}"
)

# check how much data is in the directory
if os.path.exists(new_cached_activations_path):
    print("Directory exists. Checking how much data is in the directory.")
    total_files = sum(
        os.path.getsize(os.path.join(new_cached_activations_path, f))
        for f in os.listdir(new_cached_activations_path)
        if os.path.isfile(os.path.join(new_cached_activations_path, f))
    )
    print(f"Total size of directory: {total_files / 1e9:.2f} GB")

if device == "cuda":
    torch.cuda.empty_cache()
elif device == "mps":
    torch.mps.empty_cache()


# If the directory exists, delete it.
if input("Delete the directory? (y/n): ") == "y" and os.path.exists(
    new_cached_activations_path
):
    if os.path.exists(new_cached_activations_path):
        shutil.rmtree(new_cached_activations_path)

cfg = CacheActivationsRunnerConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    hook_name=f"blocks.31.hook_resid_post",
    hook_layer=31,
    dataset_path=dataset_path,
    streaming=True,
    new_cached_activations_path=new_cached_activations_path,
    # Add these parameters:
    d_in=model_config.hidden_size,  # Set this to the correct dimension of the MLP output
    context_size=1024,  # Set an appropriate context size
    store_batch_size_prompts=16,  # Adjust as needed
    n_batches_in_buffer=32,  # Adjust as needed
    training_tokens=total_training_tokens,  # Set the total number of tokens to cache
    dtype="float32",  # Specify the dtype
    device="cuda" if torch.cuda.is_available() else "cpu",  # Set the device
    prepend_bos=True,
    normalize_activations="none",

)

start_time = time.time()

runner = CacheActivationsRunner(cfg)

print("-" * 50)
print(runner.__str__())
print("-" * 50)
runner.run()

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(
    f"{total_training_tokens / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
)
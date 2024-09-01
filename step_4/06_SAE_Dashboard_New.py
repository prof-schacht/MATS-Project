# %%
import os
import torch

from sae_dashboard.neuronpedia.neuronpedia_runner import (
    NeuronpediaRunner,
    NeuronpediaRunnerConfig,
)


# %%

# python neuronpedia.py generate --sae-set=res-jb --sae-path=/opt/Gemma-2b-Residual-Stream-SAEs/gemma_2b_blocks.10.hook_resid_post_16384 --dataset-path=Skylion007/openwebtext --log-sparsity=-6 --dtype= --feat-per-batch=128 --n-prompts=24576 --n-context-tokens=128 --n-prompts-in-forward-pass=128 --resume-from-batch=0 --end-at-batch=-1


NP_OUTPUT_FOLDER = "/proj/neuronpedia_outputs/"
ACT_CACHE_FOLDER = "/proj/cached_activations"
NP_SET_NAME = "phi3-mini-res-16L-65k"
SAE_SET = "phi3-mini-res-16L-65k"
SAE_PATH = "/proj/checkpoint_vis/final_1024000000"
NUM_FEATURES_PER_BATCH = 128
HF_DATASET_PATH = "coai/dclm-baseline-subset_100k"


SPARSITY_THRESHOLD = 1

# IMPORTANT
SAE_DTYPE = "float32"
MODEL_DTYPE = "float32"

# PERFORMANCE SETTING
# N_PROMPTS = 24576
N_PROMPTS = 24576
N_TOKENS_IN_PROMPT = 128
N_PROMPTS_IN_FORWARD_PASS = 32


#if __name__ == "__main__":

# delete output files if present
os.system(f"rm -rf {NP_OUTPUT_FOLDER}")
os.system(f"rm -rf {ACT_CACHE_FOLDER}")

# # we make two batches of 2 features each
cfg = NeuronpediaRunnerConfig(
    sae_set="local",
    sae_path=SAE_PATH,
    np_set_name=NP_SET_NAME,
    from_local_sae=True,
    huggingface_dataset_path=HF_DATASET_PATH,
    sae_dtype=SAE_DTYPE,
    model_dtype=MODEL_DTYPE,
    outputs_dir=NP_OUTPUT_FOLDER,
    sparsity_threshold=SPARSITY_THRESHOLD,
    n_prompts_total=N_PROMPTS,
    n_tokens_in_prompt=N_TOKENS_IN_PROMPT,
    n_prompts_in_forward_pass=N_PROMPTS_IN_FORWARD_PASS,
    n_features_at_a_time=NUM_FEATURES_PER_BATCH,
    start_batch=0,
    use_wandb=True,
    # sis
    #model_from_pretrained_kwargs = {'n_devices': torch.cuda.device_count() - 1},
    #model_device = "cuda",
    activation_store_device = "cpu",
    model_device = "cuda",
    model_n_devices = 2,
    sae_device = "cuda",
    # TESTING ONLY
    #end_batch=1,
)

runner = NeuronpediaRunner(cfg)
runner.run()


# %%




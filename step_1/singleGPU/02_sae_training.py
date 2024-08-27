# %%
import torch
import os
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import torch
import os

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%

total_training_steps = 100_000
batch_size = 2048
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = total_training_steps // 10
lr_decay_steps = total_training_steps
l1_warm_up_steps = total_training_steps // 20 # 5 % of training

#togethercomputer/RedPajama-Data-1T-Sample

cfg = LanguageModelSAERunnerConfig(
    # data generation function
    model_name="microsoft/Phi-3-mini-4k-instruct",
    hook_name="blocks.31.hook_resid_post",
    hook_layer=31,
    d_in=3072, # the with of of the mlp output
    dataset_path="mlfoundations/dclm-baseline-1.0", 
    is_dataset_tokenized=False,
    use_cached_activations=True,
    cached_activations_path="/proj/cached_activations/microsoft/Phi-3-mini-4k-instruct/dclm-baseline-1.0/100000",
    streaming=True,
    # SAE training parameters 
    mse_loss_normalization=None,
    expansion_factor=32, # the width if the SAE. Larger will be better but slower training
    b_dec_init_method="zeros", # the gemoetric median can be used to initialize the decoder weights, but zeros is faster
    apply_b_dec_to_input=False, 
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    # Training parameters
    lr=1e-5,
    adam_beta1=0.9, # adam_params (default but experiment later)
    adam_beta2=0.999, # adam_params (default but experiment later)
    lr_scheduler_name="constant", # constant learning rate with warmup. Could be better scheduled later.
    lr_warm_up_steps=lr_warm_up_steps, # this can help to avoid too many dead features initially
    lr_decay_steps=lr_decay_steps, # this helps us to avoid overfitting
    l1_coefficient=15, # will control how sparse the features are.
    l1_warm_up_steps=l1_warm_up_steps, # this can help to avoid too many dead features initially
    lp_norm=1.0, # the L1 penality (and not a Lp fp p < 1)
    train_batch_size_tokens=batch_size,
    context_size=1024, # will control the length of the prompts we feed to the model. Larger is better but slower 
    # Activation Store Parameter
    n_batches_in_buffer=32, # controls how many activation we store / shuffle
    training_tokens=total_training_tokens,
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False, # we don't use ghost grads anymore
    feature_sampling_window=2000, # this controls our reporting of feature aprsity stats
    dead_feature_window=1000, # would effecht resampling or ghost grads id we were using it
    dead_feature_threshold=1e-3, # would effect resampling if we were using it
    # wandb
    log_to_wandb=True, # always use wandb unless you are just testing code.
    wandb_project="phi3-mini-sae",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32"
)

sparse_autoencoder = SAETrainingRunner(cfg).run()
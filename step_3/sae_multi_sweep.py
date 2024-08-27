import torch
import argparse
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

def main(args):
    if torch.cuda.is_available():
        device = args.cuda_device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg_dict = {
        # Original parameters
        "model_name": args.model_name,
        "hook_name": args.hook_name,
        "hook_layer": args.hook_layer,
        "d_in": args.d_in,
        "dataset_path": args.dataset_path,
        "context_size": args.context_size,
        "is_dataset_tokenized": args.is_dataset_tokenized,
        "prepend_bos": True,
        "expansion_factor": args.expansion_factor,
        "training_tokens": args.total_training_tokens,
        "train_batch_size_tokens": args.batch_size,
        "mse_loss_normalization": None,
        "l1_coefficient": args.l1_coefficient,
        "lp_norm": args.lp_norm,
        "scale_sparsity_penalty_by_decoder_norm": True,
        "lr_scheduler_name": args.lr_scheduler_name,
        "l1_warm_up_steps": args.l1_warmup_steps,
        "lr_warm_up_steps": args.lr_warm_up_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "use_ghost_grads": False,
        "apply_b_dec_to_input": False,
        "b_dec_init_method": "zeros",
        "normalize_sae_decoder": False,
        "decoder_heuristic_init": True,
        "init_encoder_as_decoder_transpose": True,
        "lr": args.lr,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "n_batches_in_buffer": 64,
        "store_batch_size_prompts": 16,
        "normalize_activations": "expected_average_only_in",
        "n_eval_batches": 3,
        "eval_batch_size_prompts": 4,
        "feature_sampling_window": 1000,
        "dead_feature_window": 1000,
        "dead_feature_threshold": 1e-4,
        "compile_sae": False,
        "log_to_wandb": True,
        "wandb_project": args.wandb_project,
        "wandb_log_frequency": args.wandb_log_frequency,
        "device": args.cuda_device,
        "seed": 42,
        "n_checkpoints": args.n_checkpoints,
        "checkpoint_path": args.checkpoint_path,
        "dtype": "float32",
        
        # Additional parameters from phi3_sae_runner_multiple_devices
        "streaming": args.streaming,
        "model_from_pretrained_kwargs": {'n_devices': torch.cuda.device_count() - 1},
        "act_store_device": args.act_store_device,
        "eval_every_n_wandb_logs": args.eval_every_n_wandb_logs,
        "dataset_trust_remote_code": args.dataset_trust_remote_code,
    }

    cfg = LanguageModelSAERunnerConfig(**cfg_dict)
    sae = SAETrainingRunner(cfg).run()
    assert sae is not None

if __name__ == "__main__":

    # total_training_steps = 20_000
    total_training_steps = 40_000
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size
    print(f"Total Training Tokens: {total_training_tokens}")

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps / 5
    print(f"lr_decay_steps: {lr_decay_steps}")
    l1_warmup_steps = total_training_steps * 0.4
    print(f"l1_warmup_steps: {l1_warmup_steps}")

    parser = argparse.ArgumentParser(description="SAE Training Script")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct", help="Name of the model")
    parser.add_argument("--dataset_path", default="mlfoundations/dclm-baseline-1.0", help="Path to the dataset")
    parser.add_argument("--hook_name", default="blocks.16.hook_resid_post", help="Hook name")
    parser.add_argument("--hook_layer", type=int, default=16, help="Hook layer")
    parser.add_argument("--d_in", type=int, default=3072, help="Input dimension")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size")
    parser.add_argument("--is_dataset_tokenized", action="store_true", help="Is the dataset tokenized")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming")
    parser.add_argument("--cuda_device", default="cuda:2", help="CUDA device number")
    parser.add_argument("--act_store_device", default="cpu", help="Device for storing activations")
    parser.add_argument("--eval_every_n_wandb_logs", type=int, default=3, help="Evaluate every n wandb logs")
    parser.add_argument("--context_size", type=int, default=1024, help="Context size")
    parser.add_argument("--wandb_project", default="phi3-mini-sae-sweep", help="Wandb project name")
    parser.add_argument("--wandb_log_frequency", type=int, default=30, help="Wandb log frequency")
    parser.add_argument("--expansion_factor", type=int, default=32, help="Expansion factor")
    parser.add_argument("--dataset_trust_remote_code", action="store_true", default=True, help="Trust remote code for dataset")
    parser.add_argument("--lr_scheduler_name", default="constant", help="Learning rate scheduler name")
    parser.add_argument("--l1_coefficient", type=float, default=5, help="L1 coefficient")
    parser.add_argument("--lp_norm", type=float, default=1.0, help="Lp norm")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--total_training_tokens", type=int, default=total_training_tokens, help="Total training tokens")
    parser.add_argument("--l1_warmup_steps", type=int, default=l1_warmup_steps, help="L1 warmup steps")
    parser.add_argument("--lr_warm_up_steps", type=int, default=lr_warm_up_steps, help="LR warm up steps")
    parser.add_argument("--lr_decay_steps", type=int, default=lr_decay_steps, help="LR decay steps")
    parser.add_argument("--n_checkpoints", type=int, default=1, help="Number of checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")


    args = parser.parse_args()
    main(args)

    # python sae_multi_training.py --lr 1e-4 --expansion_factor 64 --l1_coefficient 3 --lp_norm 2.0 --lr_scheduler_name cosine
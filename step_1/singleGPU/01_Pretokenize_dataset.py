from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig

cfg = PretokenizeRunnerConfig(
    tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
    #dataset_path="NeelNanda/c4-10k", # this is just a tiny test dataset
    dataset_path="mlfoundations/dclm-baseline-1.0",
    column_name="text",
    shuffle=True, # Only not streaming
    streaming=True, # PRetokenizer_runner has to be changed not using the num_proc with an iterable dataset. 
    #num_proc=32, # increase this number depending on how many CPUs you have
    # tweak these settings depending on the model
    context_size=768,
    begin_batch_token=1, # Id for the bos token <s>
    begin_sequence_token=None,
    sequence_separator_token=32000, # Id for the bos token <|endoftext|>

    # uncomment to upload to huggingface
    hf_repo_id="coai/dclm-tokenized-phi3"

    # uncomment to save the dataset locally
    # save_path="./c4-10k-tokenized-gpt2"
)

dataset = PretokenizeRunner(cfg).run()
import yaml
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load configuration from config.yml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

base_model_name = config['base_model_name']
adapter_model_name = config['adapter_model_name']
max_shard_size = config['max_shard_size']
merged_path = config['merged_path']

# Display progress with tqdm
with tqdm(total=100, desc="Loading and Merging Models", ncols=100) as pbar:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    pbar.update(10)
    print(f"AutoTokenizer {tokenizer} loaded into memory")

    print(f"Started loading {base_model_name} into memory")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    pbar.update(50)

    model = PeftModel.from_pretrained(model, adapter_model_name)
    pbar.update(20)

    model = model.merge_and_unload()
    pbar.update(10)

    model.save_pretrained(merged_path, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(merged_path)
    pbar.update(10)

print("Model and tokenizer saved to", merged_path)


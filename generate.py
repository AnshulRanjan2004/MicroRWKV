import os
import argparse
import tiktoken
import torch
import time

from modelGenerate import GPT
from dataclasses import dataclass

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True,
                    help='Prompt for generation')
parser.add_argument('--max_num_tokens', type=int, default=100,
                    help='Maximum number of tokens to generate')
parser.add_argument('--model_name', type=str, required=True,
                    help='Name of the model checkpoint')
args = parser.parse_args()


@dataclass
class GPTConfig:
    block_size: int = 1024

    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = 50304

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768

    num_experts: int = 4
    num_active_experts: int = 4
    expert_dim: int = 512
    dim: int = 768

    dropout: float = 0.0

    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = False


# Load the model checkpoint
ckpt_path = os.path.join('./out', f'{args.model_name}.pt')
checkpoint = torch.load(ckpt_path,torch.device('cpu'))
print(checkpoint['config'])
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
# model.cuda()
model.eval()

# Encode the prompt using tiktoken
enc = tiktoken.get_encoding("gpt2")
prompt_ids = enc.encode_ordinary(args.prompt)

# Measure inference time
start_time = time.time()  # Get the current time before generating text
generated = model.generate(torch.tensor(
    [prompt_ids], device='cpu'), max_new_tokens=args.max_num_tokens)
end_time = time.time()  # Get the current time after generating text
inference_time = end_time - start_time  # Calculate inference time in seconds

# Convert seconds to more readable format
if inference_time >= 3600:
    hours = int(inference_time // 3600)
    minutes = int((inference_time % 3600) // 60)
    seconds = int(inference_time % 60)
    inference_time_str = f"{hours} hours {minutes} minutes {seconds} seconds"
elif inference_time >= 60:
    minutes = int(inference_time // 60)
    seconds = int(inference_time % 60)
    inference_time_str = f"{minutes} minutes {seconds} seconds"
else:
    seconds = int(inference_time)
    inference_time_str = f"{seconds} seconds"

output = enc.decode(generated[0].tolist())

print(f"Prompt: {args.prompt}")
print(f"Generated text: {output}")
print(f"Generated text length: {len(output)}")
print(f"Inference time: {inference_time_str}")

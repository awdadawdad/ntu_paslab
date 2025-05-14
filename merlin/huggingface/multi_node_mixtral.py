# inference.py
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

state  = PartialState()
device = state.device

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token #HEREHEREHEREHEREHEREHEREHEREHEREHEREHERE
model     = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype="auto", device_map={"": device}
)

prompts = [
    "Hello, how are you?",
    "Explain mixture-of-experts.",
    "Give me a haiku about spring.",
    "Summarize the concept of attention in transformers.",
    "how is your day"
]
print("Loading model from:", args.model_path)

with state.split_between_processes(prompts) as prompt:
    ids  = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device) #HEREHEREHEREHEREHEREHEREHEREHEREHEREHERE
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1024)
    print(f"[rank={state.process_index}] {tokenizer.decode(out[0], skip_special_tokens=True)}")
    print(f"[rank={state.process_index}] Using device: {device}, total ranks: {state.num_processes}")
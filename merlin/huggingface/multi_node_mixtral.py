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
model     = AutoModelForCausalLM.from_pretrained(
    args.model_path, torch_dtype=torch.float16, device_map={"": device}
)

prompts = ["Hello, how are you?",
           "Explain mixture-of-experts.",
           "Give me a haiku about spring."]

with state.split_between_processes(prompts) as prompt:
    ids  = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=64)
    print(f"[rank={state.process_index}] {tokenizer.decode(out[0], skip_special_tokens=True)}")

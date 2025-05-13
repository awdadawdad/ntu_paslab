# inference.py
from accelerate import PartialState                    
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

state = PartialState()                                 
device = state.device

model_name = "/mnt/disk2/llm_team/Mixtral-8x7B-Instruct-v0.1"                   
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": device},                          
)
prompts = [
    "Hello, how are you?",
    "Explain mixture-of-experts.",
    "Give me a haiku about spring."
]


with state.split_between_processes(prompts) as prompt:
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**input_ids, max_new_tokens=64)
    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[rank={state.process_index}] {txt}")

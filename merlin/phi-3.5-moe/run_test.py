import sys
import os
import torch
from transformers import AutoTokenizer, pipeline
from inference import PhiMoEForCausalLM  
from configuration_phimoe import PhiMoEConfig
from pathlib import Path


torch.random.manual_seed(0)

gpu = torch.device(f"cpu")

config = PhiMoEConfig()

with torch.device("meta"):
    model = PhiMoEForCausalLM(config)

print(model)
'''
for id in range (16):
    expert = torch.load(
            f"/mnt/disk2/llm_team/merlin_phi3.5_weights/ep/experts-{id}.pt",
            map_location=gpu,
            weights_only=True,
            mmap=True,
        )
    
    model.load_state_dict(expert, assign=True, strict=True)
''' 
'''
non_experts = torch.load(
            "/mnt/disk2/llm_team/merlin_phi3.5_weights/ep/non-experts.pt",
            map_location=gpu,
            weights_only=True,
            mmap=True,
        )

model.load_state_dict(non_experts, assign=True, strict=True)

'''
'''
tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")

prompt =  ["hello", 
           "hi", 
           "how is today"]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

generation_args = {
    "max_new_tokens": 8,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(prompt, batch_size=len(prompt), **generation_args)
for item in output:
    print(item)

'''
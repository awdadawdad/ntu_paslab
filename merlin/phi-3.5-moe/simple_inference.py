import sys
import os
import torch
from transformers import AutoTokenizer, pipeline
from inference import PhiMoEForCausalLM  


torch.random.manual_seed(0)


model = PhiMoEForCausalLM.from_pretrained(
    "/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,  
)

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
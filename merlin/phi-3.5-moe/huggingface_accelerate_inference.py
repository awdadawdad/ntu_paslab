import sys
import os
import torch
from transformers import AutoTokenizer, pipeline
from inference import PhiMoEForCausalLM
from accelerate import Accelerator

accelerator = Accelerator()

torch.random.manual_seed(0)


model = PhiMoEForCausalLM.from_pretrained(
    "/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,  
)

model = accelerator.prepare(model)

tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")

prompt = "给我写一个关于ai技术的文章"

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(prompt, **generation_args)
print(output[0]['generated_text'])
print("Attention implementation:", model.config._attn_implementation)

import torch
from transformers import AutoTokenizer, pipeline
from inference import PhiMoEForCausalLM  



model = PhiMoEForCausalLM.from_pretrained(
    "/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
    device_map="auto",
    torch_dtype="float32",
    trust_remote_code=False,  
)

tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")

prompt =  ["给我讲讲中国十二生肖好吗？"]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(prompt, **generation_args)

print(output)
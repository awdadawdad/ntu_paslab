import torch
from transformers import AutoTokenizer, pipeline
from inference import PhiMoEForCausalLM  



model = PhiMoEForCausalLM.from_pretrained(
    "/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,  
)

tokenizer = AutoTokenizer.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")
print(tokenizer.pad_token_id)
prompt =  ["how is dog"
           ]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

generation_args = {
    "max_new_tokens": 2,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(prompt, batch_size= 1,**generation_args)

print(output)
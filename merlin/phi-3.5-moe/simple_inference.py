import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


torch.random.manual_seed(0) 


model = AutoModelForCausalLM.from_pretrained( 
    "/mnt/llm_team/Phi-3.5-MoE-instruct",  
    device_map="auto",      
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("/mnt/llm_team/Phi-3.5-MoE-instruct") 



prompt = "Hey, are you conscious? Can you talk to me?"


pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device_map = "auto"
) 


generation_args = { 
    "max_new_tokens": 128, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 


output = pipe(prompt, **generation_args) 
print(output[0]['generated_text'])

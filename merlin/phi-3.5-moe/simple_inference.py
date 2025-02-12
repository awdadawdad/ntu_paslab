import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 


torch.random.manual_seed(0) 


model = AutoModelForCausalLM.from_pretrained( 
    "/mnt/llm_team/Phi-3.5-MoE-instruct",  
    device_map="auto",      
    torch_dtype="auto",  
    trust_remote_code=False,  
) 

tokenizer = AutoTokenizer.from_pretrained("/mnt/llm_team/Phi-3.5-MoE-instruct") 



messages = [
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    }
]


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


output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])

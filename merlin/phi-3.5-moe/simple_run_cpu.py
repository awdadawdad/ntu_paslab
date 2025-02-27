import torch
from transformers import AutoTokenizer
from raw_model import PhiMoEForCausalLM
torch.random.manual_seed(0)


model = PhiMoEForCausalLM.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-moe-instruct")

prompt = "讲讲中国十二生肖是什么"
inputs = tokenizer(prompt, return_tensors="pt")


generate_ids = model.generate(inputs.input_ids, max_length=1024)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
import torch
from transformers import AutoTokenizer
from modeling_phimoe import PhiMoEForCausalLM
from safetensors.torch import load_file, save_file


#weight = load_file("/mnt/disk2/llm_team/merlin_phi3.5_weights/test/model.safetensors")
model = PhiMoEForCausalLM.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
                                          #state_dict = weight,
                                          torch_dtype =  "auto",
                                          device_map = "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-moe-instruct")

prompt = "给我讲讲中国十二生肖是怎么回事。"
inputs = tokenizer(prompt, return_tensors="pt")


generate_ids = model.generate(inputs.input_ids, max_length = 512)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
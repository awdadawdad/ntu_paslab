import torch
from modeling_phimoe import PhiMoEForCausalLM
from safetensors.torch import load_file, save_file
from pathlib import Path




model = PhiMoEForCausalLM.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct",
                                          torch_dtype =  "auto")
model.eval()

torch.save(model.state_dict(), "/mnt/disk2/llm_team/merlin_phi3.5_weights/only_config/model.pt")
'''

model_path = Path("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")
all_weight = {}
for file_path in model_path.glob("*.safetensors"):
    
    state_dict = load_file(file_path)
#for key, tensor in weight.items():
    #print(f"{key}: {tensor.dtype}")
# 加载权重
    all_weight.update(state_dict)
save_file(all_weight,"/mnt/disk2/llm_team/merlin_phi3.5_weights/test/model.safetensors")
'''
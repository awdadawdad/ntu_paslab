import torch

from raw_model import PhiMoEForCausalLM




model = PhiMoEForCausalLM.from_pretrained("/mnt/disk2/llm_team/Phi-3.5-MoE-instruct")
model.eval()

torch.save(model, "/mnt/disk2/llm_team/merlin_phi3.5_weights/test/model.pt")
from vllm import LLM, SamplingParams   # LLM is originally imported from here
from transformers import AutoTokenizer
import torch

import json


def add_template(tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    res = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        txt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        res.append(txt)
    return res


torch.manual_seed(7)
model_path = "/mnt/disk2/llm_team/silicon_mind/QwQ-32B"
prompt_path = "/home/gaven/ntu_paslab/merlin/prompts/diverse_short.json"
n_prompts = 1
bsz = 1

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    enforce_eager=False,
    # max_num_seqs=2,
    max_model_len=140,
    enable_chunked_prefill=True,
    #gpu_memory_utilization=0.98  # ✅ 新增这一行
)
with open(prompt_path, "r") as f:
    prompts = add_template(tokenizer, json.load(f)["prompts"][:n_prompts])

for i in range(0, n_prompts, bsz):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs, performance = llm.generate(prompts[i:i+bsz], sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")

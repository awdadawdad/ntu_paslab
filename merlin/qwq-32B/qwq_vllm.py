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
prompt_path = "/mnt/disk3/gaven/ntu_paslab/merlin/prompts/128.json"
n_prompts = 4
bsz = 2

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,
    enforce_eager=False,
    max_num_seqs=bsz,
    max_model_len=256,
    enable_chunked_prefill=False,
    gpu_memory_utilization=0.75,
    max_num_batched_tokens = 65536,
)
with open(prompt_path, "r") as f:
    prompts = add_template(tokenizer, json.load(f)["prompts"][:n_prompts])

for i in range(0, n_prompts, bsz):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate(prompts[i:i+bsz], sampling_params)

    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt}, Generated text: {generated_text}")

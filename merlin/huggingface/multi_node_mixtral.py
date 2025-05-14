# inference.py
from accelerate import PartialState,inference
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--pp_size", type=int, default=4) 
args = parser.parse_args()

state  = PartialState()
device = state.device

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    torch_dtype="auto", 
    #device_map="auto"
)

pp_model = inference.prepare_pippy(
    model,
    pipeline_parallel_size=args.pp_size,   # 4 stage = 4 GPU
    kwargs={"use_cuda_rpc": True},         # 多节点必开
)


prompts = ["Hello, how are you?"]

with state.split_between_processes(prompts) as subset:
    for p in subset:
        ids = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = pp_model.generate(**ids, max_new_tokens=64)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if state.is_main_process:
            print(text, flush=True)

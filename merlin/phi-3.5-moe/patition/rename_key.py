import torch
import argparse
def rename_keys_and_save(old_ckpt_path: str, new_ckpt_path: str):
   
    state_dict = torch.load(old_ckpt_path, weights_only=True, mmap=True)

    new_state_dict = {}

    for old_key, value in state_dict.items():
       
        new_key = old_key

        if old_key.startswith("model.layers."):
            
            new_key = new_key.replace("model.layers.", "layers.")

            new_key = new_key.replace(".self_attn.", ".attention.")

            new_key = new_key.replace(".block_sparse_moe.", ".feed_forward.")

            new_key = new_key.replace(".input_layernorm.", ".attention_norm.")

            new_key = new_key.replace(".post_attention_layernorm.", ".ffn_norm.")

            
        if old_key.startswith("lm_head."):
            new_key = new_key.replace("lm_head.", "output.")
            
        if old_key.startswith("model.embed_tokens."):
            new_key = new_key.replace("model.embed_tokens.", "embed_tokens.")
        
        if old_key.startswith("model.norm."):
            new_key = new_key.replace("model.norm.", "norm.")
        
        
        new_state_dict[new_key] = value

    torch.save(new_state_dict, new_ckpt_path)
    print(f"finish {new_ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    rename_keys_and_save(args.model_path, args.output_path)


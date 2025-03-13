"""
Partitioner designer for mixtral-8x7b weights (original code, minimal modification)
Only does expert parallel, no tensor parallel, no non-expert splitting.
"""

from pathlib import Path
import argparse
import glob
import json
import logging
import torch
import fnmatch


class Partitioner:

    def __init__(self, model_path: str, output_path: str) -> None:
        self.model_path = Path(model_path)
        self.output_path = Path(output_path) if output_path else self.model_path

       
        self.model_config = (
            self.get_configs()
        )

    def get_configs(self) -> dict:
        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)

        

        return model_config

    def load_weights(self) -> dict:
        weight = torch.load(self.model_path/"model.pt",
                            
                            weights_only=True,
                            mmap=True,)
          
        return weight

    def layers(self, ws: dict, num_gpu: int) -> dict:
        num_layers = self.model_config["num_hidden_layers"]

        dicts = [{} for _ in range(num_gpu)]
        
        all_keys = [{} for _ in range(num_layers)]
        for li in range(num_layers):

            pattern = f"model.layers.{li}.*"
              
            for key in list(ws.keys()):
                if fnmatch.fnmatch(key, pattern):
                    value = ws.pop(key)
                    all_keys[li][key] = value

        step = num_layers // num_gpu
        for i in range (num_gpu):
            for j in range (step):
                dicts[i].update(all_keys[i*step + j])
            dicts[i]["model.norm.bias"] = ws["model.norm.bias"]
            dicts[i]["model.norm.weight"] = ws["model.norm.weight"]
            torch.save(dicts[i], self.output_path / f"gpu{i}.pt")
        
        return ws

        
        

    def last_layer(self, ws: dict) -> None:
        ws.pop("model.norm.weight")
        ws.pop("model.norm.bias")
        embed_tokens = {}
        value = ws.pop("model.embed_tokens.weight")
        embed_tokens["model.embed_tokens.weight"] = value
        torch.save(embed_tokens, self.output_path / f"embed_tokens.pt")
        torch.save(ws, self.output_path / f"lm_head.pt")
        

    def start(self) -> None:
        ws = self.layers(self.load_weights(), 8)
        logging.info("finished partitioning layers weights")
        self.last_layer(ws)
        logging.info("finished partitioning last layer weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    weights_partitioner = Partitioner(
        args.model_path, args.output_path
    )
    weights_partitioner.start()
 

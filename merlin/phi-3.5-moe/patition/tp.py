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

    def partition_expert_weights(self, ws: dict, num_gpu: int) -> dict:
        num_layers = self.model_config["num_hidden_layers"]
        num_experts = self.model_config["num_local_experts"]

        dicts = [{} for _ in range(num_gpu)]
        for experts in range(num_experts):
            
            for li in range(num_layers):
                w1: torch.Tensor = ws.pop(f"model.layers.{li}.block_sparse_moe.experts.{experts}.w1.weight")
                w2: torch.Tensor = ws.pop(f"model.layers.{li}.block_sparse_moe.experts.{experts}.w2.weight")
                w3: torch.Tensor = ws.pop(f"model.layers.{li}.block_sparse_moe.experts.{experts}.w3.weight")
 
                
                w1 = torch.chunk(w1, num_gpu, dim = 0)
                print(len(w1))
                w2 = torch.chunk(w2, num_gpu, dim = 1)
                print(len(w2))
                w3 = torch.chunk(w3, num_gpu, dim = 0)
                print(len(w3))
                print("     ")
                for i in range (num_gpu):
                    dicts[i][f"layer.{li}.expert.{experts}.w1"] = w1[i].clone()
                    dicts[i][f"layer.{li}.expert.{experts}.w2"] = w2[i].clone()
                    dicts[i][f"layer.{li}.expert.{experts}.w3"] = w3[i].clone()

            
        for i in range (num_gpu):
            torch.save(dicts[i], self.output_path / f"gpu{i}.pt")
        return ws  


    def partition_non_expert_weights(self, ws: dict) -> None:
        
        torch.save(ws, self.output_path / f"non-experts.pt")

    def start(self) -> None:
        ws = self.partition_expert_weights(self.load_weights(), 8)
        logging.info("finished partitioning experts weights")
        self.partition_non_expert_weights(ws)
        logging.info("finished partitioning non-experts weights")


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


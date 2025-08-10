import torch

from cs336_basics.modules import Transformer
from cs336_basics.utils import complete


def main():
    config_dir = "llm_train/logs/tinystories/config.json"
    model_dir = "llm_train/logs/tinystories/model_Aug_10.bin"
    out_path = "llm_train/tinystories_samples.txt"

    model = Transformer.from_config(config_dir, device="cuda:0")
    sd = torch.load(model_dir, map_location="cuda:0")

    model.load_state_dict(sd["model"])
    out = complete(model, prefix="Once upon a time", num_choices=32, max_length=256, device="cuda:0")
    with open(out_path, "w") as f:
        for s in out:
            f.write(s)
            f.write("<|endofgen|>\n\n")


if __name__ == "__main__":
    main()

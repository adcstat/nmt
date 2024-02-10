"""
This script processes and aggregates training and validation losses stored in checkpoint files.
"""
import argparse
from collections import defaultdict
import os
import numpy as np
import torch

def collect_losses(losses_path: str):
    """
    Collects and processes training and validation losses from checkpoint files.

    Args:
        losses_path (str): The directory containing the checkpoint files.
    """
    train_losses = []
    val_losses = []
    fps = [os.path.join(losses_path, fn) for fn in os.listdir(losses_path) if fn.endswith(".tar")]
    fps.sort()

    # Group files by checkpoint (excluding the last part after the last underscore)
    checkpoint_groups = defaultdict(list)
    for fp in fps:
        base_name = os.path.basename(fp)
        checkpoint_key = base_name.rsplit("_", maxsplit=1)[0]
        checkpoint_groups[checkpoint_key].append(fp)
    
    for checkpoint_key, files in checkpoint_groups.items():
        for fp in files:
            checkpoint = torch.load(fp)
            train_losses.extend(np.array(checkpoint["TRAIN_LOSSES"]).T.mean(-1)) # average over GPUs
        val_losses.append(np.array(torch.load(files[-1])["VAL_LOSSES"]).mean(-1))
        print(f"processed {fp}")

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    with open(f"{losses_path}/train_losses.npy", "wb") as fp:
        np.save(fp, train_losses)
    with open(f"{losses_path}/val_losses.npy", "wb") as fp:
        np.save(fp, val_losses)
    print("saved losses")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    collect_losses(args.path)

if __name__ == "__main__":
    main()

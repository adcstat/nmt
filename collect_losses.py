import argparse
import os
import numpy as np
import torch

def collect_losses(checkpoint_path):
    train_losses = []
    val_losses = []
    fps = [os.path.join(checkpoint_path, fn) for fn in os.listdir(checkpoint_path)]
    fps.sort()
    for fp in fps:
        checkpoint = torch.load(fp)
        train_losses.append(checkpoint["TRAIN_LOSSES"])
        val_losses.append(checkpoint["VAL_LOSS"])
        print(f"processed {fp}")
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    with open(f"{checkpoint_path}/train_losses.npy", "wb") as fp:
        np.save(fp, train_losses)
    with open(f"{checkpoint_path}/val_losses.npy", "wb") as fp:
        np.save(fp, val_losses)

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("path", type=str, help="Path to the checkpoint directory")
    args = parser.parse_args()
    collect_losses(args.path, args.epochs)

if __name__ == "__main__":
    main()

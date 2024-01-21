import argparse
import numpy as np
import torch

def collect_losses(checkpoint_path, num_checkpoints):
    train_losses = []
    val_losses = []
    for i in range(1, num_checkpoints + 1):
        checkpoint = torch.load(f"{checkpoint_path}/checkpoint_{i}.tar")
        train_losses.append(checkpoint["TRAIN_LOSSES"])
        val_losses.append(checkpoint["VAL_LOSS"])
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    with open(f"{checkpoint_path}/train_losses.npy", "wb") as fp:
        np.save(fp, train_losses)
    with open(f"{checkpoint_path}/val_losses.npy", "wb") as fp:
        np.save(fp, val_losses)

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint directory")
    parser.add_argument("number_of_checkpoints", type=int, help="Number of checkpoints to process")
    args = parser.parse_args()
    collect_losses(args.checkpoint_path, args.number_of_checkpoints)

if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import torch

def collect_losses(losses_path):
    train_losses = []
    val_losses = []
    fps = [os.path.join(losses_path, fn) for fn in os.listdir(losses_path) if fn.endswith(".tar")]
    fps.sort()
    for fp in fps:
        checkpoint = torch.load(fp)
        train_losses.extend(np.array(checkpoint["TRAIN_LOSSES"]).T.mean(-1))
        val_losses.append(checkpoint["VAL_LOSSES"])
        print(f"processed {fp}")
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    with open(f"{losses_path}/train_losses.npy", "wb") as fp:
        np.save(fp, train_losses)
    with open(f"{losses_path}/val_losses.npy", "wb") as fp:
        np.save(fp, val_losses)
    print("saved losses")

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("--config", required=True, type=str, help="config of experiment to use")
    args = parser.parse_args()
    collect_losses(f"checkpoints/{args.config}")

if __name__ == "__main__":
    main()

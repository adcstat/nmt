import numpy as np
import torch

train_losses = []
val_losses = []
for i in range(1, 6):
    checkpoint = torch.load(f"checkpoints/checkpoint_{i}.tar")
    train_losses.append(checkpoint["TRAIN_LOSSES"])
    val_losses.append(checkpoint["VAL_LOSS"])
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

with open("checkpoints/train_losses.npy", "wb") as fp:
    np.save(fp, train_losses)
with open("checkpoints/val_losses.npy", "wb") as fp:
    np.save(fp, val_losses)
import os
import json
import numpy as np
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tokenizers import Tokenizer

from utils.data_utils import get_dataloader, BatchedDataset
from utils import transformer_utils as tfu

with open("params.json", "r") as fp:
    params = json.load(fp)

with open("checkpoints/params.json", "w") as fp:
    json.dump(params, fp)

PAD_IDX = 2
vocab_size = params["vocab_size"]
tokens_per_batch = params["tokens_per_batch"]
epochs = params["epochs"]
grad_accumulation = params["grad_accumulation"]
d_model = params["d_model"]
n_heads = params["n_heads"]
d_ff = params["d_ff"]
n_layers = params["n_layers"]
dropout = params["dropout"]

tokenizer = Tokenizer.from_file("data/bpe_tokenizer.json")

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.model = model.to(self.gpu_id)
        self.train_data = train_data

        self.train_data_len = len(self.train_data)
        self.opt_steps_per_epoch = self.train_data_len // grad_accumulation
        self.steps_till_print = 500
        self.val_data = val_data
        self.val_data_len = len(self.val_data)
        self._print_infos()

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.scaler = torch.cuda.amp.GradScaler()
        self.schedule = tfu.TransformerScheduler(
            self.optimizer,
            warmup_steps=int(0.5 * self.opt_steps_per_epoch), # first half epoch is warmup
            max_rate=0.0015
        )

        self.epochs_run = 0
        self.snapshot_path = "checkpoints/snapshot.tar"
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _print_infos(self):
        if self.gpu_id == 0:
            print(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6}M parameters")
            print(f"Training data has {self.train_data_len} batches")
            print(f"There are {self.opt_steps_per_epoch} opt steps per epoch")

    def _save_snapshot(self, epoch, train_losses, val_loss):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULE_STATE": self.schedule.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSSES": train_losses,
            "VAL_LOSS": val_loss,
        }
        # save snapshot for reloading
        torch.save(snapshot, self.snapshot_path)
        # save snapshot for checkpoint averaging
        torch.save(snapshot, f"checkpoints/checkpoint_{epoch}.tar")
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        print("********************************************")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.schedule.load_state_dict(snapshot["SCHEDULE_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch):
        self.model.train()
        losses = np.array([])
        self.train_data.sampler.set_epoch(epoch)
        for batch_i, (src, tgt) in enumerate(self.train_data):
            batch_i += 1
            loss = self._run_batch(src, tgt, batch_i)
            losses = np.append(losses, loss)
            if (batch_i % self.steps_till_print == 0) and (self.gpu_id == 0):
                print(f"Loss since last {self.steps_till_print} steps (batch {batch_i} of {self.train_data_len}; opt step {batch_i // grad_accumulation} of {self.opt_steps_per_epoch}): ", losses[-self.steps_till_print:].sum() / self.steps_till_print)
        return losses

    def _run_batch(self, src, tgt, batch_i):
        src = src.to(self.gpu_id)
        tgt = tgt.to(self.gpu_id)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].reshape(-1)

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_in == PAD_IDX

        with torch.cuda.amp.autocast():
            logits = self.model(src, tgt_in, src_padding_mask, tgt_padding_mask).reshape(-1, vocab_size)
            loss = self.loss_fn(logits, tgt_out)

        self.scaler.scale(loss).backward()
        # Gradient Accumulation
        if batch_i % grad_accumulation == 0:
          # Scales loss and performs backward pass using automatic mixed precision
            self.scaler.step(self.optimizer)
            self.schedule.step()
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return np.array([loss.item()])

    def _evaluate(self):
        self.model.eval()
        losses = 0
        for src, tgt in self.val_data:
            src = src.to(self.gpu_id)
            tgt = tgt.to(self.gpu_id)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:].reshape(-1)

            src_padding_mask = src == PAD_IDX
            tgt_padding_mask = tgt_in == PAD_IDX

            logits = self.model(src, tgt_in, src_padding_mask, tgt_padding_mask).reshape(-1, vocab_size)
            loss = self.loss_fn(logits, tgt_out)

            losses += loss.item()

        return losses / self.val_data_len

    def train(self):
        for epoch in range(self.epochs_run+1, epochs+1):
            start_time = timer()
            train_losses = self._run_epoch(epoch)
            duration = timer() - start_time
            val_loss = self._evaluate()
            print(f"[GPU{self.gpu_id}] epoch duration: {duration} | val_loss: {val_loss}")
            all_train_losses = [None for _ in range(self.world_size)] if self.gpu_id == 0 else None
            all_val_losses = [None for _ in range(self.world_size)] if self.gpu_id == 0 else None
            dist.gather_object(train_losses, all_train_losses, dst=0)
            dist.gather_object(val_loss, all_val_losses, dst=0)
            if self.gpu_id == 0:
                self._save_snapshot(epoch, all_train_losses, all_val_losses)


def main():
    ddp_setup()
    # easily fits into memory
    with open("data/wmt14.json", "r") as fp:
        wmt14 = json.load(fp)
    train_data = get_dataloader(BatchedDataset(wmt14["train"], tokens_per_batch), tokenizer)
    val_data = get_dataloader(BatchedDataset(wmt14["validation"], tokens_per_batch), tokenizer)
    model = tfu.Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
        dropout=dropout
    )
    trainer = Trainer(model, train_data, val_data)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
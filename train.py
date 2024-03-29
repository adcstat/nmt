"""
Executes distributed training of a Transformer model with PyTorch, loading model configurations,
training, and validating across multiple GPUs. It supports dynamic parameter loading, checkpointing,
and utilizes a custom learning rate scheduler.
"""
import importlib
import argparse
import os
import json
import numpy as np
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer

from utils.data_utils import get_dataloader, BatchedDataset
from utils import decoding_utils

PAD_IDX = 2

def set_global_params(param_config: str, model_config: str) -> None:
    """
    Sets global parameters for the model and training based on configuration files.

    Args:
        param_config (str): Identifier for the parameter configuration.
        model_config (str): Identifier for the model configuration.
    """
    with open(f"params/params_{param_config}.json", "r") as fp:
        params = json.load(fp)

    global checkpoint_dir
    checkpoint_dir = f"checkpoints/{vocab_size}/{model_config}/{param_config}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    global tokenizer, tokens_per_batch, epochs, tokens_per_opt_step, d_model, n_heads, d_ff, n_layers, dropout, warmup_steps, max_lr
    tokenizer = Tokenizer.from_file(f"data/{vocab_size}/bpe_tokenizer.json")
    tokens_per_batch = params["tokens_per_batch"]
    epochs = params["epochs"]
    tokens_per_opt_step = params["tokens_per_opt_step"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    d_ff = params["d_ff"]
    n_layers = params["n_layers"]
    dropout = params["dropout"]
    warmup_steps = params["warmup_steps"]
    max_lr = params["max_lr"]

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements a learning rate scheduler with a warmup phase where the learning rate increases linearly
    from 0 to a maximum rate, followed by inverse squareroot decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): The number of steps over which the learning rate will warm up.
        max_rate (float): The maximum learning rate after warmup.

    The learning rate at step `s` is `lr = max_rate * min((warmup_steps / s)^0.5, s / warmup_steps)`.
    """
    def __init__(self, optimizer, warmup_steps, max_rate):
        self.warmup_steps = warmup_steps
        self.max_rate = max_rate
        super().__init__(optimizer)

    def get_lr(self):
        step_num = self._step_count
        lr = self.max_rate * min((self.warmup_steps / step_num)**0.5, step_num / self.warmup_steps)
        return [lr for _ in self.optimizer.param_groups]

class Trainer:
    """
    Manages the training process for a Transformer model including logging, checkpointing, and evaluation.

    Args:
        model (torch.nn.Module): The Transformer model to train.
        train_data (DataLoader): DataLoader for the training dataset.
        val_data (DataLoader): DataLoader for the validation dataset.

    The class is responsible for managing the training loop, handling gradient accumulation, applying
    gradient scaling for mixed precision training, updating the learning rate, logging losses to TensorBoard,
    and saving/loading checkpoints.
    """
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
        self.writer = SummaryWriter(f"{checkpoint_dir}/tb")

        self.train_data_len = len(self.train_data)
        self.grad_accumulation = tokens_per_opt_step // (self.world_size * tokens_per_batch)
        self.opt_steps_per_epoch = self.train_data_len // self.grad_accumulation
        self.checkpoints_per_epoch = 5
        self.start_checkpoint_at_epoch = 15
        self.steps_till_print = 500
        self.val_data = val_data
        self.val_data_len = len(self.val_data)
        self._print_infos()

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.scaler = torch.cuda.amp.GradScaler()
        self.schedule = LRScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            max_rate=max_lr
        )

        self.epochs_run = 0
        self.snapshot_path = f"{checkpoint_dir}/snapshot.tar"
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _print_infos(self) -> None:
        """
        Prints information about the model and training setup to the console.
        """
        if self.gpu_id == 0:
            print(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6}M parameters")
            print(f"Training data has {self.train_data_len} batches")
            print(f"There are {self.opt_steps_per_epoch} opt steps per epoch")

    def _save_snapshot(self, epoch: int, cp: int, train_losses: np.ndarray, val_losses: np.ndarray) -> None:
        """
        Saves a snapshot of the model's state, optimizer's state, learning rate scheduler's state,
        and losses for each epoch and checkpoint.
        """
        os.makedirs(f"{checkpoint_dir}/losses", exist_ok=True)
        losses = {
            "TRAIN_LOSSES": train_losses,
            "VAL_LOSSES": val_losses
        }
        epoch_str = str(epoch) if epoch >= 10 else f"0{epoch}"
        torch.save(losses, f"{checkpoint_dir}/losses/losses_{epoch_str}_{cp}.tar")
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULE_STATE": self.schedule.state_dict(),
            "EPOCHS_RUN": epoch,
            "CHECKPOINT": cp,
        }
        # save snapshot every full epoch for reloading
        if cp == self.cpe_running:
            torch.save(snapshot, self.snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
            print("-----------------------------------------")
        if epoch >= self.start_checkpoint_at_epoch:
            # save checkpoint for checkpoint averaging
            cp_path = f"{checkpoint_dir}/checkpoint_{epoch}_{cp}.tar"
            torch.save(snapshot, cp_path)
            print(f"Epoch {epoch} cp {cp} | Checkpoint saved at {cp_path}")
            print("********************************************")

    def _load_snapshot(self, snapshot_path: str) -> None:
        """
        Loads model, optimizer, and scheduler states from a snapshot to resume training.
        """
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.schedule.load_state_dict(snapshot["SCHEDULE_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, src: torch.Tensor, tgt: torch.Tensor, batch_i: int) -> np.ndarray:
        """
        Processes a single batch of data, computes the loss, and performs a backward pass.
        """
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
        if batch_i % self.grad_accumulation == 0:
            # Scales loss and performs backward pass using automatic mixed precision
            self.scaler.step(self.optimizer)
            self.schedule.step()
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return np.array([loss.item()])

    def _evaluate(self) -> float:
        """
        Evaluates the model on the validation dataset and returns the average loss.
        """
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

    def _test_translate(self) -> None:
        """
        Translates a set of predefined sentences to visually inspect the model's translation quality.
        """
        if self.gpu_id == 0:
            test_sentences = [
                "Birds fly in the sky.",
                "The large dog is barking loudly.",
                "She painted the wall blue yesterday.",
                "They have not visited the new museum.",
                "If it snows, we will go skiing tomorrow.",
                "Before the sun sets, the children play in the park, under the big oak tree.",
                "The cake, which was baked by my grandmother, was eaten at the party.",
                "If I were elected president, I would implement major reforms.",
                "She must finish her report before she can leave, even though she prefers to do it tomorrow.",
                "Despite the burgeoning apprehensions, the dedicated linguists, meticulously analyzing archaic manuscripts, endeavored to decipher the intricate symbology inherent in the ancient texts."
            ]
            for sentence in test_sentences:
                translation = decoding_utils.translate(tokenizer, self.model.module, sentence, 4, self.gpu_id)
                print(f"src: {sentence} \ntranslation: {translation}")

    def train(self) -> None:
        """
        Executes the training loop, periodically evaluating the model and saving checkpoints.
        """
        print("lr: ", self.optimizer.param_groups[0]["lr"])
        self._test_translate()
        self.model.train()
        for epoch in range(self.epochs_run+1, epochs+1):
            self.cpe_running = 1 if epoch < self.start_checkpoint_at_epoch else self.checkpoints_per_epoch
            losses = np.array([])
            start_time = timer()
            self.train_data.sampler.set_epoch(epoch)
            for batch_i, (src, tgt) in enumerate(self.train_data):
                batch_i += 1
                loss = self._run_batch(src, tgt, batch_i)
                losses = np.append(losses, loss)
                if self.gpu_id == 0:
                    self.writer.add_scalar('Training Loss', loss, epoch * self.train_data_len + batch_i)
                if (self.cpe_running == 1) and (batch_i % self.steps_till_print == 0) and (self.gpu_id == 0):
                    print("lr: ", self.optimizer.param_groups[0]["lr"])
                    print(f"Loss since last {self.steps_till_print} steps (batch {batch_i} of {self.train_data_len}; opt step {batch_i // self.grad_accumulation} of {self.opt_steps_per_epoch}): ", losses[-self.steps_till_print:].sum() / self.steps_till_print)
                if batch_i % (self.train_data_len // self.cpe_running) == 0:
                    cp = batch_i // (self.train_data_len // self.cpe_running)
                    val_loss = self._evaluate()
                    self._test_translate()
                    self.model.train()
                    duration = timer() - start_time
                    print(f"[GPU{self.gpu_id}] epoch {epoch} cp {cp}; duration: {duration}; train_loss {losses.mean()}; val_loss: {val_loss}")
                    all_train_losses = [None for _ in range(self.world_size)] if self.gpu_id == 0 else None
                    all_val_losses = [None for _ in range(self.world_size)] if self.gpu_id == 0 else None
                    dist.gather_object(losses, all_train_losses, dst=0)
                    dist.gather_object(val_loss, all_val_losses, dst=0)
                    if self.gpu_id == 0:
                        self._save_snapshot(epoch, cp, all_train_losses, all_val_losses)
                    losses = np.array([])
                    start_time = timer()
        self.writer.flush()
        self.writer.close()
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-v", required=True, type=int, help="size of vocabulary")
    parser.add_argument("--param_config", "-pc", required=True, type=str, help="param config of experiment to use")
    parser.add_argument("--model_config", "-mc", required=True, type=str, help="model config of experiment to use")
    args = parser.parse_args()

    global vocab_size, tfu
    vocab_size = args.vocab_size
    tfu = importlib.import_module(f"utils.transformer_utils_{args.model_config}")

    set_global_params(args.param_config, args.model_config)
    ddp_setup()
    # easily fits into memory
    with open(f"data/{vocab_size}/wmt14.json", "r") as fp:
        wmt14 = json.load(fp)
    train_data = get_dataloader(BatchedDataset(wmt14["train"], tokens_per_batch), tokenizer)
    val_data = get_dataloader(BatchedDataset(wmt14["validation"], tokens_per_batch // 2), tokenizer)
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
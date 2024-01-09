# python standard
import json
import numpy as np
from timeit import default_timer as timer

# huggingface
from tokenizers import Tokenizer

# torch
from torch.utils.data import DataLoader
from torch import Tensor
import torch
import torch.nn as nn

# custom
from utils import transformer_utils as tfu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("drive/MyDrive/nmt/data_list.json", "r") as fp:
    data_list = json.load(fp)
    
# Define special symbols and indices
BOS_IDX, EOS_IDX, PAD_IDX = 0, 1, 2
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_tokens = ['<bos>', '<eos>', '<pad>']

tokenizer = Tokenizer.from_file("drive/MyDrive/nmt/bpe_tokenizer.json")

# function to batch data into batches of certain token size
def batch_data_fn(data, max_tokens):
    batched_data = []
    batch = []
    batch_length = 0
    for src, tgt, length in data:
        if batch_length + length <= max_tokens:
            batch.append([src, tgt])
            batch_length += length
        else:
            batched_data.append(batch)
            batch = []
            batch_length = 0
    return batched_data

# function to collate data samples into batch tensors
def collate_fn(batch):
    batch_transposed = np.asarray(batch).T
    src_batch, tgt_batch = batch_transposed[0], batch_transposed[1]
    src_batch = tokenizer.encode_batch(src_batch)
    tgt_batch = tokenizer.encode_batch(tgt_batch)
    src_batch = torch.tensor([encoding.ids for encoding in src_batch])
    tgt_batch = torch.tensor([encoding.ids for encoding in tgt_batch])
    return src_batch, tgt_batch


d_model = 512
n_heads = 8
d_ff = 2 * d_model
n_layers = 6
dropout = 0.1
vocab_size = tokenizer.get_vocab_size()
transformer = tfu.Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_encoder_layers=n_layers,
    n_decoder_layers=n_layers,
    dropout=dropout
)
transformer = transformer.to(DEVICE)

max_tokens = 6000
batched_train_data = batch_data_fn(data_list["train"], max_tokens)
batched_val_data = batch_data_fn(data_list["validation"], max_tokens)
train_dataloader = DataLoader(batched_train_data, batch_size=None, collate_fn=collate_fn, pin_memory=True, num_workers=2)
val_dataloader = DataLoader(batched_val_data, batch_size=None, collate_fn=collate_fn, pin_memory=True, num_workers=2)
train_dataloader_len = len(train_dataloader)
val_dataloader_len = len(val_dataloader)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
optimizer = torch.optim.AdamW(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
scaler = torch.cuda.amp.GradScaler()

epochs = 15
grad_accumulation = 16
steps = epochs * train_dataloader_len // grad_accumulation
warumup_steps = int(0.04 * steps)
warmup_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warumup_steps)
cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps-warumup_steps, eta_min=0.0001)
schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_schedule, cosine_schedule], milestones=[warumup_steps])


def train_epoch(model):
    model.train()
    losses = np.array([])
    for batch_i, (src, tgt) in enumerate(train_dataloader):
        batch_i += 1
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].reshape(-1)

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_in == PAD_IDX

        with torch.cuda.amp.autocast():
            logits = model(src, tgt_in, src_padding_mask, tgt_padding_mask).reshape(-1, vocab_size)
            loss = loss_fn(logits, tgt_out)

        scaler.scale(loss).backward()
        # Gradient Accumulation
        if batch_i % grad_accumulation == 0:
          # Scales loss and performs backward pass using automatic mixed precision
          scaler.step(optimizer)
          schedule.step()
          scaler.update()
          optimizer.zero_grad(set_to_none=True)

        losses = np.append(losses, np.array([loss.item()]))
        if batch_i % (grad_accumulation * 100) == 0:
            print(f"accumulated loss so far (batch {batch_i}; opt step {batch_i / grad_accumulation}): ", losses.sum() / batch_i)


@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:].reshape(-1)

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_in == PAD_IDX

        logits = transformer(src, tgt_in, src_padding_mask, tgt_padding_mask).reshape(-1, vocab_size)
        loss = loss_fn(logits, tgt_out)

        losses += loss.item()

    return losses / val_dataloader_len


for epoch in range(1, epochs+1):
    start_time = timer()
    losses = train_epoch(transformer)
    duration = timer() - start_time
    val_loss = evaluate(transformer)

    torch.save(
        {
          'model_state_dict': transformer.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'schedule_state_dict': schedule.state_dict(),
          'loss': losses,
          "val_loss": val_loss,
          "duration": duration
        },
        f"drive/MyDrive/nmt/transformer_epoch_{epoch}.tar"
    )
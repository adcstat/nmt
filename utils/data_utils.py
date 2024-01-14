import json
import numpy as np

from tokenizers import Tokenizer

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import sys
sys.path.append("..")

tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

def get_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

class BatchedDataset(Dataset):
    def __init__(self, data, tokens_per_batch):
        self.data = batch_data_fn(data, tokens_per_batch)
        self.size = len(self.data)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return self.data[index]

# function to batch data into batches of certain token size
def batch_data_fn(data, tokens_per_batch):
    batched_data = []
    batch = []
    batch_length = 0
    for src, tgt, length in data:
        if batch_length + length <= tokens_per_batch:
            batch.append([src, tgt])
            batch_length += length
        else:
            batched_data.append(batch)
            batch = []
            batch_length = 0
            batch.append([src, tgt])
            batch_length += length
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

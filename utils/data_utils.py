"""
This module is used to provide dataloaders of data batched by token count
"""
from typing import List, Tuple
import numpy as np
from tokenizers import Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

def get_dataloader(dataset: Dataset, tokenizer: Tokenizer) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=None,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

class BatchedDataset(Dataset):
    """
    A Dataset that groups data into batches of a specific token count.

    Args:
        data (List[Tuple[str, str, int]]): List of tuples containing source text, target text, and length.
        tokens_per_batch (int): The maximum number of tokens per batch.
    """
    def __init__(self, data, tokens_per_batch):
        self.data = batch_data_fn(data, tokens_per_batch)
        self.size = len(self.data)
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return self.data[index]

def batch_data_fn(data: List[Tuple[str, str, int]], tokens_per_batch: int) -> List[List[List[str]]]:
    """
    Organizes data into batches where the total token count does not exceed a specified limit.

    Args:
        data (List[Tuple[str, str, int]]): The dataset to be batched, consisting of tuples of (source, target, length).
        tokens_per_batch (int): Maximum number of tokens in each batch.

    Returns:
        List[List[List[str]]]: Data organized into batches.
    """
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

def collate_fn(batch: List[List[str]], tokenizer: Tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collates batches of raw text data into tensors.

    Args:
        batch (List[List[str]]): A batch of data to be collated.
        tokenizer: The tokenizer used for encoding text data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors for source and target batches.
    """
    batch_transposed = np.asarray(batch).T
    src_batch, tgt_batch = batch_transposed[0], batch_transposed[1]
    src_batch = tokenizer.encode_batch(src_batch)
    tgt_batch = tokenizer.encode_batch(tgt_batch)
    src_batch = torch.tensor([encoding.ids for encoding in src_batch])
    tgt_batch = torch.tensor([encoding.ids for encoding in tgt_batch])
    return src_batch, tgt_batch

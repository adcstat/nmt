"""
This module provides methods to load the wmt14 en-de data and train a byte-level BPE tokenizer
"""
import argparse
import os
import json

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as bl_pre
from tokenizers.processors import ByteLevel as bl_post, Sequence, TemplateProcessing
from tokenizers.decoders import ByteLevel as bl_dec


BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2
special_tokens = ["<bos>", "<eos>", "<pad>"]

def load_data():
    """
    Loads the WMT14 English-German dataset.

    Returns:
        dict: A dictionary containing 'train', 'validation', and 'test' splits of the dataset.
    """
    data_iter = load_dataset("wmt14", 'de-en')
    data_dict = {split: [[item["translation"]["en"], item["translation"]["de"]] for item in data_iter[split]] for split in ["train", "validation", "test"]}
    return data_dict

def flatten_data(data: list) -> list:
    """
    Flattens a list of translation pairs to prepare for tokenizer training.

    Args:
        data (list): A list of translation pairs.

    Returns:
        list: A flattened list of all text data.
    """
    text_list = []
    for item in data:
        text_list.extend(item)
    return text_list

def initialize_tokenizer() -> Tokenizer:
    """
    Initializes a Byte-Pair Encoding (BPE) tokenizer with specific pre and post-processing configurations.

    Returns:
        Tokenizer: A tokenizer object ready for training.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = bl_pre(add_prefix_space=False)
    add_bos_eos = TemplateProcessing(
        single=f"{special_tokens[BOS_IDX]} $A {special_tokens[EOS_IDX]}",
        special_tokens=[(special_tokens[BOS_IDX], BOS_IDX), (special_tokens[EOS_IDX], EOS_IDX)],
    )
    tokenizer.post_processor = Sequence([bl_post(trim_offsets=False), add_bos_eos])
    tokenizer.decoder = bl_dec()
    tokenizer.enable_padding(pad_id=PAD_IDX, pad_token=special_tokens[PAD_IDX])
    return tokenizer

def train_tokenizer(tokenizer: Tokenizer, text_list: list, vocab_size: int) -> None:
    """
    Trains the tokenizer on the provided text list.

    Args:
        tokenizer (Tokenizer): The tokenizer to be trained.
        text_list (list): The list of text data for training.
        vocab_size (int): The desired vocabulary size.
    """
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens, show_progress=True)
    tokenizer.train_from_iterator(text_list, trainer)
    tokenizer.save(f"data/{vocab_size}/bpe_tokenizer.json")

def process_data(tokenizer: Tokenizer, data_dict: dict, max_length: int) -> dict:
    """
    Processes the data to enforce a maximum sequence length and sorts entries by sequence length.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for encoding the sequences.
        data_dict (dict): The dataset split dictionary.
        max_length (int): The maximum token count for sequences.

    Returns:
        dict: A dictionary of processed dataset splits.
    """
    tokenizer.no_padding()
    processed_data = {}
    for split in data_dict:
        data = data_dict[split]
        # calculate token count
        data_src = []
        data_tgt = []
        for src, tgt in data:
            data_src.append(src)
            data_tgt.append(tgt)

        data_src_enc = tokenizer.encode_batch(data_src)
        data_tgt_enc = tokenizer.encode_batch(data_tgt)

        data_src_lengths = [len(item.ids) for item in data_src_enc]
        data_tgt_lengths = [len(item.ids) for item in data_tgt_enc]

        max_lengths = [max(src_length, tgt_length) for src_length, tgt_length in zip(data_src_lengths, data_tgt_lengths)]
        with_lengths = list(zip(data_src, data_tgt, max_lengths))

        # sort by token count decreasing
        sorted_data = sorted(with_lengths, key=lambda x: x[2], reverse=True)
        # clip length of sentence to avoid extremely uneven batches
        if split != "test":
            sorted_data = [[src, tgt, length] for src, tgt, length in sorted_data if length <= max_length]
        processed_data[split] = sorted_data
    return processed_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", "-v", required=True, type=int, help="size of vocabulary")
    parser.add_argument("--max_length", "-ml", required=True, type=int, help="maximum token count of sequences in data")
    args = parser.parse_args()
    vocab_size = args.vocab_size
    max_length = args.max_length
    data_dict = load_data()
    print("loaded data!")
    train_text_list = flatten_data(data_dict["train"])
    tokenizer = initialize_tokenizer()
    print("initialized tokenizer!")
    os.makedirs(f"data/{vocab_size}", exist_ok=True)
    train_tokenizer(tokenizer, train_text_list, vocab_size)
    print("trained tokenizer!")
    processed_data = process_data(tokenizer, data_dict, max_length)
    print("processed data!")
    with open(f"data/{vocab_size}/wmt14.json", "w") as fp:
        json.dump(processed_data, fp)
    with open(f"data/{vocab_size}/wmt14_validation.json", "w") as fp:
        json.dump(processed_data["validation"], fp)
    with open(f"data/{vocab_size}/wmt14_test.json", "w") as fp:
        json.dump(processed_data["test"], fp)
    print("saved data!")

if __name__ == "__main__":
    main()

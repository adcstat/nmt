import json

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as bl_pre
from tokenizers.processors import ByteLevel as bl_post, Sequence, TemplateProcessing
from tokenizers.decoders import ByteLevel as bl_dec


with open("params.json", "r") as fp:
    params = json.load(fp)

BOS_IDX = params["BOS_IDX"]
EOS_IDX = params["EOS_IDX"]
PAD_IDX = params["PAD_IDX"]
special_tokens = params["special_tokens"]
vocab_size = params["vocab_size"]
max_length = params["max_length"]


def load_data():
    data_iter = load_dataset("wmt14", 'de-en')
    data = {split: [[item["translation"]["en"], item["translation"]["de"]] for item in data_iter[split]] for split in ["train", "validation", "test"]}
    return data

def flatten_data(data):
    # make flattened list of all strings for bpe trainer
    text_list = []
    for item in data:
        text_list.extend(item)
    return text_list

def initialize_tokenizer():
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

def train_tokenizer(tokenizer, text_list):
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens, show_progress=True)
    tokenizer.train_from_iterator(text_list, trainer)
    tokenizer.save("bpe_tokenizer.json")

def process_data(tokenizer, data):
    processed_data = {}
    for split in data:
        # calculate token count
        with_lengths = [[src, tgt, max(len(tokenizer.encode(src).ids), len(tokenizer.encode(tgt).ids))] for src, tgt in data[split]]
        # sort by token count decreasing
        sorted_data = sorted(with_lengths, key=lambda x: x[2], reverse=True)
        # clip length of sentence to avoid extremely uneven batches
        if split != "test":
            sorted_data = [[src, tgt, length] for src, tgt, length in sorted_data if length <= max_length]
        processed_data[split] = sorted_data
    return processed_data


def main():
    data = load_data()
    print("loaded data!")
    train_text_list = flatten_data(data["train"])
    tokenizer = initialize_tokenizer()
    print("initialized tokenizer!")
    train_tokenizer(tokenizer, train_text_list)
    print("trained tokenizer!")
    processed_data = process_data(tokenizer, data)
    print("processed data!")
    data.update(processed_data)
    with open("wmt14.json", "w") as fp:
        json.dump(data, fp)
    print("saved data!")

if __name__ == "__main__":
    main()
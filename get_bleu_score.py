# python standard
import importlib
import os
import json
import csv
import argparse

# huggingface
from tokenizers import Tokenizer

# torch
import torch
from torch.utils.data import DataLoader

# custom
from utils import decoding_utils, data_utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(path, checkpoint):
    with open(f"{path}/params.json", "r") as fp:
        params = json.load(fp)

    tokens_per_batch = params["tokens_per_batch"]
    vocab_size = params["vocab_size"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    d_ff = params["d_ff"]
    n_layers = params["n_layers"]
    dropout = params["dropout"]

    model = tfu.Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
        dropout=dropout
    )
    model = model.to(DEVICE)

    checkpoint = torch.load(f"{path}/{checkpoint}.tar")
    model.load_state_dict(checkpoint['MODEL_STATE'])

    return model, tokens_per_batch

def load_test_data(split, tokens_per_batch, tokenizer):
    with open("data/wmt14_200.json", "r") as fp:
        test_data = json.load(fp)
    test_data_batched = data_utils.BatchedDataset(test_data[split], tokens_per_batch)
    test_dataloader = DataLoader(
        test_data_batched,
        batch_size=None,
        collate_fn=lambda batch: data_utils.collate_fn(batch, tokenizer),
        pin_memory=True,
        shuffle=False
    )
    return test_dataloader

def save_bleu(config, split, beam_width, bleu):
    bleu_file_path = 'checkpoints/bleus.csv'
    # Check if file exists to decide if we need to write headers
    file_exists = os.path.exists(bleu_file_path)
    
    with open(bleu_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write headers if the file is being created for the first time
        if not file_exists:
            writer.writerow(["Config", "Split", "Beam Width", "BLEU Score"])
        # Write the data
        writer.writerow([config, split, beam_width, bleu])
    
    print(f"BLEU score written to {bleu_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("--config", required=True, type=str, help="config of experiment to use")
    parser.add_argument("--checkpoint", required=True, type=str, help="name of model state to use")
    parser.add_argument("--split", required=True, type=str, help="which split to use")
    parser.add_argument("--beam_width", required=True, type=int, help="beam_width")
    args = parser.parse_args()

    config = args.config
    checkpoint = args.checkpoint
    split = args.split
    beam_width = args.beam_width

    global tfu
    tfu = importlib.import_module(f"utils.transformer_utils_{config}")
    
    tokenizer = Tokenizer.from_file(f"data/bpe_tokenizer.json")
    path = f"checkpoints/{config}"
    model, tokens_per_batch  = load_model(path, checkpoint)
    test_dataloader = load_test_data(split, tokens_per_batch, tokenizer)

    bleu = decoding_utils.get_bleu_score(tokenizer, model, test_dataloader, beam_width, DEVICE).item()
    print(bleu)
    save_bleu(config, split, beam_width, bleu)

if __name__ == "__main__":
    main()
# python standard
import json
import argparse

# huggingface
from tokenizers import Tokenizer

# torch
import torch
from torch.utils.data import DataLoader

# custom
from utils import decoding_utils, data_utils, transformer_utils as tfu
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
    test_data = [[src, tgt] for src, tgt, _ in test_data[split]]
    test_data_batched = data_utils.BatchedDataset(test_data, tokens_per_batch)
    test_dataloader = DataLoader(
        test_data_batched,
        batch_size=None,
        collate_fn=lambda batch: data_utils.collate_fn(batch, tokenizer),
        pin_memory=True,
        shuffle=False
    )
    return test_dataloader

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("--config", required=True, type=str, help="config of experiment to use")
    parser.add_argument("--checkpoint", required=True, type=int, help="name of model state to use")
    parser.add_argument("--split", required=True, type=int, help="which split to use")
    parser.add_argument("--beam_width", required=True, type=int, help="beam_width")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(f"data/bpe_tokenizer.json")
    model, tokens_per_batch  = load_model(f"checkpoints/{args.config}", args.checkpoint)
    test_dataloader = load_test_data(args.split, tokens_per_batch, tokenizer)

    bleu = decoding_utils.get_bleu_score(tokenizer, model, test_dataloader, args.beam_width, DEVICE)
    print(bleu.item())


if __name__ == "__main__":
    main()
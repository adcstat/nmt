"""
Analyzes Transformer attention weights by calculating and plotting entropy distributions.
"""
import importlib
from typing import List, Tuple
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tokenizers import Tokenizer

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from utils import data_utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_IDX = 2

def load_model_and_data(vocab_size: int, param_config: str, model_config: str, checkpoint: str):
    """
    Loads the Transformer model and validation data based on the specified configurations.

    Args:
        vocab_size (int): Vocabulary size used for the tokenizer.
        param_config (str): Parameter configuration file name.
        model_config (str): Model configuration identifier.
        checkpoint (str): Checkpoint file name to load the model weights.

    Returns:
        tuple: Loaded model, DataLoader for the validation dataset.
    """
    with open(f"params/params_{param_config}.json", "r") as fp:
        params = json.load(fp)

    tokens_per_batch = params["tokens_per_batch"]
    tokenizer = Tokenizer.from_file(f"data/{vocab_size}/bpe_tokenizer.json")
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

    global checkpoint_dir
    checkpoint_dir = f"checkpoints/{vocab_size}/{model_config}/{param_config}"
    checkpoint = torch.load(f"{checkpoint_dir}/{checkpoint}.tar")
    model.load_state_dict(checkpoint['MODEL_STATE'])

    with open(f"data/{vocab_size}/wmt14_validation.json", "r") as fp:
        test_data = json.load(fp)
    test_data_batched = data_utils.BatchedDataset(test_data, tokens_per_batch)
    test_dataloader = DataLoader(
        test_data_batched,
        batch_size=None,
        collate_fn=lambda batch: data_utils.collate_fn(batch, tokenizer),
        pin_memory=True,
        shuffle=False
    )

    return model, test_dataloader

def gather_attn_weights(model: torch.nn.Module, test_dataloader: DataLoader):
    """
    Gathers attention weights from the model for the given dataset.

    Args:
        model (torch.nn.Module): The loaded Transformer model.
        test_dataloader (DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: Encoder attention weights, decoder self-attention weights, encoder-decoder attention weights.
    """
    enc_attn_batch_list = []
    dec_self_batch_list = []
    enc_dec_batch_list = []
    for src, tgt in test_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_in == PAD_IDX

        enc_att_weights_all, dec_self_att_weights_all, enc_dec_weights_all = model.get_attention_weights(src, tgt_in, src_padding_mask, tgt_padding_mask)
        enc_attn_batch_list.append(enc_att_weights_all) # (n_batches, n_layers, n_heads, batch_size, seq_len_q, seq_len_kv)
        dec_self_batch_list.append(dec_self_att_weights_all)
        enc_dec_batch_list.append(enc_dec_weights_all)
    return enc_attn_batch_list, dec_self_batch_list, enc_dec_batch_list

def calculate_entropy(tensor: Tensor) -> Tensor:
    """
    Calculates entropy for attention weight distributions.
    
    Args:
        tensor (torch.Tensor): A tensor of attention weights.

    Returns:
        torch.Tensor: The calculated entropy of the attention weights.
    """
    # Replace zeros with a small number to avoid log(0)
    tensor = torch.clamp(tensor, min=1e-9)
    inter = -tensor * torch.log(tensor)
    entropy = inter.sum(-1)
    return entropy

def calc_entropy_batch_list(attn_batch_list: List[Tensor]) -> torch.Tensor:
    """
    Aggregates and calculates entropy for batches of attention weights.
    
    Args:
        attn_batch_list (List[torch.Tensor]): A list of tensors representing attention weights for multiple batches.

    Returns:
        torch.Tensor: Aggregated entropy values for each attention head across batches.
    """
    n_batches = len(attn_batch_list)
    n_layers = len(attn_batch_list[0])
    n_heads = len(attn_batch_list[0][0])
    entropy_list = []
    for layer in range(n_layers):
        head_list = []
        for head in range(n_heads):
            batch_list = []
            for batch in range(n_batches):
                batch_list.append(calculate_entropy(attn_batch_list[batch][layer][head]).flatten()) # (batch_size * seq_len)
            batch_list_cat = torch.cat(batch_list) # (n_batches * batch_size * avg_seq_len)
            batch_list_cat = batch_list_cat[~batch_list_cat.isnan()]
            head_list.append(batch_list_cat)
        entropy_list.append(torch.stack(head_list))
    return torch.stack(entropy_list) # (n_layers, n_heads, n_batches * batch_size * avg_seq_len)

def sort_by_median(entropy_tensor: Tensor) -> Tuple[Tensor, float, float]:
    """
    Sorts attention heads based on median entropy values.
    
    Args:
        entropy_tensor (torch.Tensor): A tensor containing entropy values for attention heads.

    Returns:
        Tuple containing the sorted tensor and the lower and upper quantile values for entropy.
    """
    medians = entropy_tensor.median(-1).values # (n_layers, n_heads)
    lower_quantile, upper_quantile = np.percentile(medians.flatten().cpu(), [33, 66])
    # sort medians by heads for each layer
    sorted_indices = torch.argsort(medians, dim=1).unsqueeze(-1).expand(-1, -1, entropy_tensor.shape[-1])
    sorted_tensor = torch.gather(entropy_tensor, 1, sorted_indices)
    return sorted_tensor, lower_quantile, upper_quantile

def assign_color(median: float, lower_quantile: float, upper_quantile: float) -> str:
    """
    Determines the color for plotting based on the median's quantile.
    
    Args:
        median (float): The median value of the entropy.
        lower_quantile (float): The lower quantile of the entropy distribution.
        upper_quantile (float): The upper quantile of the entropy distribution.

    Returns:
        str: The color associated with the median's quantile.
    """
    if median <= lower_quantile:
        return 'blue'
    elif median <= upper_quantile:
        return 'yellow'
    else:
        return 'red'

def plot_entropy(entropy_tensor: torch.Tensor, lower_quantile: float, upper_quantile: float) -> plt.Figure:
    """
    Plots the entropy distributions of attention heads.
    
    Args:
        entropy_tensor (torch.Tensor): A tensor of sorted entropy values for attention heads.
        lower_quantile (float): The lower quantile value for the entropy distribution.
        upper_quantile (float): The upper quantile value for the entropy distribution.

    Returns:
        matplotlib.figure.Figure: The figure containing the plotted distributions.
    """
    # Set up the matplotlib figure
    n_layers = entropy_tensor.shape[0]
    n_heads = entropy_tensor.shape[1]
    fig, axes = plt.subplots(nrows=n_layers, ncols=n_heads, figsize=(12, 10), sharex=True)

    # Plotting each violin plot
    for layer in reversed(range(n_layers)):
        for head in range(n_heads):
            data = entropy_tensor[layer][head].cpu().numpy()
            color = assign_color(np.nanmedian(data), lower_quantile, upper_quantile)
            sns.violinplot(data=data, ax=axes[n_layers-1-layer, head], color=color, orient="h")
            if layer == n_layers-1:
                axes[n_layers-1-layer, head].set_title(f"Head: {head+1}")
            if head == 0:
                axes[n_layers-1-layer, head].set_ylabel(f"Layer {layer+1}")
            if layer == 0:
                axes[n_layers-1-layer, head].set_xlabel(f"Entropy")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Load checkpoints and save loss arrays.")
    parser.add_argument("--vocab_size", "-v", required=True, type=int, help="size of vocabulary")
    parser.add_argument("--param_config", "-pc", required=True, type=str, help="param config of experiment to use")
    parser.add_argument("--model_config", "-mc", required=True, type=str, help="model config of experiment to use")
    parser.add_argument("--checkpoint", "-c", required=True, type=str, help="name of model state to use")
    args = parser.parse_args()

    vocab_size = args.vocab_size
    param_config = args.param_config
    model_config = args.model_config
    checkpoint = args.checkpoint

    global tfu
    tfu = importlib.import_module(f"utils.transformer_utils_{model_config}")
    
    model, test_dataloader = load_model_and_data(vocab_size, param_config, model_config, checkpoint)
    print("loaded model and data")
    # get attention weights
    enc_attn_batch_list, dec_self_batch_list, enc_dec_batch_list = gather_attn_weights(model, test_dataloader)
    print("gathered attention weights")
    # get entropies
    enc_entropy = calc_entropy_batch_list(enc_attn_batch_list)
    dec_entropy = calc_entropy_batch_list(dec_self_batch_list)
    enc_dec_entropy = calc_entropy_batch_list(enc_dec_batch_list)
    print("calculated entropies")
    # sort entropies
    enc_entropy, enc_lower_quantile, enc_upper_quantile = sort_by_median(enc_entropy)
    dec_entropy, dec_lower_quantile, dec_upper_quantile = sort_by_median(dec_entropy)
    enc_dec_entropy, enc_dec_lower_quantile, enc_dec_upper_quantile = sort_by_median(enc_dec_entropy)
    print("sorted entropies")

    entropies_dict = {
        "encoder_self": [enc_entropy, enc_lower_quantile, enc_upper_quantile],
        "decoder_self": [dec_entropy, dec_lower_quantile, dec_upper_quantile],
        "encoder_decoder": [enc_dec_entropy, enc_dec_lower_quantile, enc_dec_upper_quantile]
    }

    os.makedirs(f"{checkpoint_dir}/entropy_plots", exist_ok=True)
    for entropy in entropies_dict:
        plot_entropy(*entropies_dict[entropy])
        plt.savefig(f"{checkpoint_dir}/entropy_plots/{entropy}.png", bbox_inches="tight", dpi=300)
        print(f"saved {entropy} plots")

if __name__ == "__main__":
    main()
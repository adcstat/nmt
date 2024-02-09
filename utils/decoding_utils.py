"""
This module provides utilities to decode sequences and calculate bleu scores
"""
from typing import List, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional.text import sacre_bleu_score
from tokenizers import Tokenizer

BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2

@torch.no_grad()
def beam_search_batch(
    model: torch.nn.Module,
    src: Tensor,
    beam_width: int,
    device: torch.device,
    max_len: int = 20,
    length_norm_exp: float = 0.6
) -> Tensor:
    """
    Performs beam search decoding for a batch of sequences using a given model.

    Args:
        model (torch.nn.Module): The sequence-to-sequence model used for decoding.
        src (torch.Tensor): The batch of source sequences.
        beam_width (int): The number of sequences to consider at each step (beam size).
        device (torch.device): The device on which to perform computations.
        max_len (int, optional): The maximum length of the generated sequences.
        length_norm_exp (float, optional): The exponent used for length normalization.

    Returns:
        torch.Tensor: The batch of decoded sequences with the highest probabilities.
    """
    model.eval()
    batch_size = src.shape[0]
    src_padding_mask = src == PAD_IDX
    enc = model.encode(src, src_padding_mask)
    # initialize decoding seq
    y = torch.ones(batch_size, 1, device=device).fill_(BOS_IDX).type(torch.long) # (batch_size, 1)
    # get decoding for last last token in Y
    dec = model.decode(y, enc, y == PAD_IDX, src_padding_mask)[:, -1, :] # (batch_size, d_model)
    # get logits for next predicted token
    logits = model.unembedding(dec) # (batch_size, tgt_vocab_size)
    vocabulary_size = logits.shape[-1]
    # search for beam_width highest normed probabilites within each batch
    next_probabilities, next_tokens = logits.log_softmax(-1).topk(k = beam_width, axis = -1) # (batch_size, beam_width)
    # make form (sample_1, sample_1, ...., sample_n)
    y = y.repeat((beam_width, 1)) # (batch_size*beam_width, 1)
    y = torch.cat((y, next_tokens.flatten().unsqueeze(dim=1)), dim=-1) # (batch_size*beam_width, 2)
    next_probabilities = next_probabilities.flatten().unsqueeze(1).repeat(1, vocabulary_size) # (batch_size*beam_width, tgt_vocab_size)
    src = src.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=1) # (batch_size*beam_width, src_seq_len)
    src_padding_mask = src == PAD_IDX
    enc = model.encode(src, src_padding_mask) # (batch_size*beam_width, src_seq_len, d_model)
    completed_mask = torch.zeros(batch_size * beam_width, device=device, dtype=torch.bool)

    for _ in range(max_len-1):
        # create mask to avoid updating probs for already finished seqs
        completed_mask = (y[..., -1] == EOS_IDX) | (y[..., -1] == PAD_IDX)
        if completed_mask.all():
            break
        dec = model.decode(y, enc, y == PAD_IDX, src_padding_mask)[:, -1, :] # (batch_size*beam_width, d_model)
        next_logits = model.unembedding(dec) # (batch_size*beam_width, tgt_vocab_size)
        # adding log probs instead multiplying probs
        next_probabilities[~completed_mask] += next_logits.log_softmax(-1)[~completed_mask]
        # length normalization
        lengths = torch.zeros(batch_size * beam_width, device=device).fill_(y.shape[1]) - (y == PAD_IDX).sum(dim=-1).reshape(batch_size * beam_width) # (batch_size * beam_width)
        lengths_long = lengths.unsqueeze(-1).repeat(1, vocabulary_size) # (batch_size * beam_width, vocabulary_size)
        normed_probabilities = next_probabilities / lengths_long**length_norm_exp

        # only add pad for already finished seqs
        normed_probabilities_inter = normed_probabilities
        completed_mask_long = completed_mask.unsqueeze(-1).repeat(1, vocabulary_size)
        completed_mask_long[completed_mask, PAD_IDX] = False
        normed_probabilities_inter = normed_probabilities_inter.masked_fill(completed_mask_long, float('-inf'))

        # search for beam_width highest normed probabilites within each batch
        normed_top_probabilities, idx = normed_probabilities_inter.reshape(batch_size, -1).topk(k = beam_width, axis = -1) # (batch_size, beam_width)
        top_probabilities = normed_top_probabilities * lengths.reshape(batch_size, beam_width)**length_norm_exp
        # update to top probabilities
        next_probabilities = top_probabilities.flatten().unsqueeze(-1).repeat(1, vocabulary_size)

        # calculate which tokens correspond to the highest probs
        next_tokens = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1) # (batch_size*beam_width, 1)
        # lookup which current candidates correspond to the highest probs
        best_candidates = idx // vocabulary_size # (batch_size, beam_width)
        # add correct starting index for batches
        best_candidates += torch.arange(batch_size, device=device).unsqueeze(-1) * beam_width
        # choose correct beam paths from y
        y = y[best_candidates.flatten()]
        # concat next tokens
        y = torch.cat((y, next_tokens), axis = 1)

    y = y.reshape(batch_size, beam_width, -1) # (batch_size, beam_width, max_len)
    return y[:, 0, :]

def get_bleu_score(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    beam_width: int,
    device: torch.device,
    return_preds: bool = False
) -> Union[float, Tuple[float, List[str]]]:
    """
    Calculates the BLEU score for a set of predictions from a model against reference translations.

    Args:
        tokenizer: The tokenizer used for encoding source sentences and decoding target sequences.
        model: The model to evaluate.
        test_dataloader: Dataloader containing the test dataset.
        beam_width: The beam width used for decoding.
        device: The device on which to perform computations.
        return_preds: Whether to return the predictions alongside the BLEU score.

    Returns:
        The calculated BLEU score, and optionally, the list of predicted sequences.
    """
    preds_all = []
    tgt_all = []
    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tokenizer.decode_batch(tgt.tolist())
        tgt_all.extend(tgt)

        preds = beam_search_batch(
            model=model,
            src=src,
            beam_width=beam_width,
            device=device,
            max_len=src.shape[1] + 200,
        )
        preds = tokenizer.decode_batch(preds.tolist())
        preds_all.extend(preds)
    score = sacre_bleu_score(preds_all, [[tgt_item] for tgt_item in tgt_all], smooth=True)
    if return_preds:
        return score, preds_all
    return score

@torch.no_grad()
def beam_search_single(
    model: torch.nn.Module,
    src: torch.Tensor,
    beam_width: int,
    device: torch.device,
    length_norm_exp: float = 0.7
) -> torch.Tensor:
    """
    Performs beam search decoding for a single sequence using a given model.
    I used this straight forward approach to test if the batch method is correct

    Args:
        model: The sequence-to-sequence model used for decoding.
        src: The source sequence.
        beam_width: The number of sequences to consider at each step (beam size).
        device: The device on which to perform computations.
        length_norm_exp: The exponent used for length normalization.

    Returns:
        The decoded sequence with the highest probability.
    """
    model.eval()
    src_padding_mask = src == PAD_IDX
    memory = model.encode(src, src_padding_mask)
    start_token = torch.ones(1, 1, device=device).fill_(BOS_IDX).type(torch.long)

    # Start the beam with the start token
    beam = [(start_token, 0)]  # List of tuples (sequence, score)

    for _ in range(src.shape[1] + 5):
        candidates = []
        for seq, score in beam:
            if seq[-1, -1].item() == EOS_IDX:  # Check for the end of sentence token
                candidates.append((seq, score))
                continue

            tgt_padding_mask = seq == PAD_IDX
            out = model.decode(seq, memory, tgt_padding_mask, src_padding_mask)
            prob = model.unembedding(out[:, -1, :]).log_softmax(-1)
            topk_probs, topk_indices = torch.topk(prob, beam_width, dim=-1)

            # Expand the beam with the top k next words
            for i in range(beam_width):
                next_token = topk_indices[0][i].unsqueeze(0).unsqueeze(0)
                next_score = score + topk_probs[0][i].item()    # adding log probs == multiplying probs
                candidate_seq = torch.cat([seq, next_token], dim=-1)
                candidates.append((candidate_seq, next_score))

        # Sort all candidates by score and keep the best ones
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

        if all(seq[-1, -1].item() == EOS_IDX for seq, _ in beam):
            break

    # Apply length normalization to final sequences
    normalized_beam = [(seq, score / len(seq)**(length_norm_exp)) for seq, score in beam]

    # Choose the sequence with the highest score
    best_sequence, _ = max(normalized_beam, key=lambda x: x[1])
    return best_sequence.flatten()

def translate(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    src_sentence: str,
    beam_width: int,
    device: torch.device
) -> str:
    """
    Translates a single source sentence into the target language using beam search decoding.

    Args:
        tokenizer: The tokenizer used for encoding the source sentence and decoding the generated sequence.
        model: The sequence-to-sequence model used for translation.
        src_sentence: The source sentence to translate.
        beam_width: The beam width used for decoding.
        device: The device on which to perform computations.

    Returns:
        The translated sentence.
    """
    src = torch.tensor(tokenizer.encode(src_sentence).ids, device=device).unsqueeze(dim=0)
    tgt_tokens = beam_search_batch(
        model=model,
        src=src,
        beam_width=beam_width,
        device=device,
        max_len=src.shape[1] + 50
    )[0]
    return tokenizer.decode(tgt_tokens.tolist())

def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    max_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Decodes a sequence using a greedy approach, selecting the highest probability token at each step.

    Args:
        model: The sequence-to-sequence model used for decoding.
        src: The source sequence.
        max_len: The maximum length of the generated sequence.
        device: The device on which to perform computations.

    Returns:
        The greedily decoded sequence.
    """
    model.eval()
    src_padding_mask = src == PAD_IDX
    memory = model.encode(src, src_padding_mask)
    ys = torch.ones(1, 1, device=device).fill_(BOS_IDX).type(torch.long)
    tgt_padding_mask = ys == PAD_IDX
    for _ in range(max_len-1):
        out = model.decode(ys, memory, tgt_padding_mask, src_padding_mask)
        prob = model.unembedding(out[:, -1:, :])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1, device=device).type_as(src.data).fill_(next_word)], dim=1)
        tgt_padding_mask = ys == PAD_IDX
        if next_word == EOS_IDX:
            break
    return ys.flatten()

def translate_greedy(
    tokenizer: Tokenizer,
    model: torch.nn.Module,
    src_sentence: str,
    device: torch.device
) -> str:
    """
    Translates a single source sentence into the target language using greedy decoding.

    Args:
        tokenizer: The tokenizer used for encoding the source sentence and decoding the generated sequence.
        model: The sequence-to-sequence model used for translation.
        src_sentence: The source sentence to translate.
        device: The device on which to perform computations.

    Returns:
        The translated sentence.
    """
    src = torch.tensor(tokenizer.encode(src_sentence.rstrip("\n")).ids, device=device).unsqueeze(dim=0)
    tgt_tokens = greedy_decode(
        model=model,
        src=src,
        max_len=src.shape[1] + 5,
        device=device
    )
    return tokenizer.decode(tgt_tokens.tolist())

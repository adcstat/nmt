import torch
from torchmetrics.functional.text import sacre_bleu_score


@torch.no_grad()
def beam_search(
    model,
    X,
    beam_width,
    start_symbol,
    pad_symbol,
    end_symbol,
    device,
    max_len = 20,
    only_best: bool = False,
    length_norm_exp: float = 0.6
):
    model.eval()
    batch_size = X.shape[0]
    Y = torch.ones(batch_size, 1, device=device).fill_(start_symbol).type(torch.long) # (batch_size, 1)
    src_padding_mask = X == pad_symbol
    tgt_padding_mask = Y == pad_symbol
    enc = model.encode(X, src_padding_mask)
    # get decoding for last last token in Y
    dec = model.decode(Y, enc, tgt_padding_mask, src_padding_mask)[:, -1:, :] # (batch_size, 1, d_model)
    # get logits for next predicted token
    logits = model.unembedding(dec).squeeze(1) # (batch_size, tgt_vocab_size)
    vocabulary_size = logits.shape[-1]
    # search for highest beam_width probabilites within each batch
    next_probabilities, next_tokens = logits.log_softmax(-1).topk(k = beam_width, axis = -1) # (batch_size, beam_width)
    # make form (sample_1, sample_1, ...., sample_n)
    Y = Y.repeat((beam_width, 1)) # (batch_size*beam_width, 1)
    Y = torch.cat((Y, next_tokens.flatten().unsqueeze(dim=1)), dim=-1) # (batch_size*beam_width, 2)
    next_probabilities = next_probabilities.flatten().unsqueeze(1).repeat(1, vocabulary_size) # (batch_size*beam_width, tgt_vocab_size)
    X = X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=1) # (batch_size*beam_width, src_seq_len)
    src_padding_mask = X == pad_symbol
    enc = model.encode(X, src_padding_mask) # (batch_size*beam_width, src_seq_len, d_model)
    for _ in range(max_len - 1):
        tgt_padding_mask = Y == pad_symbol
        dec = model.decode(Y, enc, tgt_padding_mask, src_padding_mask)[:, -1:, :] # (batch_size*beam_width, 1, d_model)
        next_logits = model.unembedding(dec).squeeze(1) # (batch_size*beam_width, tgt_vocab_size)
        # adding log probs instead multiplying probs
        next_probabilities += next_logits.log_softmax(-1)
        # search for highest beam_width probabilites within each batch
        top_probabilities, idx = next_probabilities.reshape(batch_size, -1).topk(k = beam_width, axis = -1) # (batch_size, beam_width)
        # update to top probabilities
        next_probabilities = top_probabilities.flatten().unsqueeze(-1).repeat(1, vocabulary_size)
        # calculate which tokens correspond to the highest probs
        next_tokens = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1) # (batch_size*beam_width, 1)
        # lookup which current candidates correspond to the highest probs
        best_candidates = idx // vocabulary_size # (batch_size, beam_width)
        # add correct starting index for batches
        best_candidates += torch.arange(batch_size, device=device).unsqueeze(-1) * beam_width
        # choose correct beam paths from Y
        Y = Y[best_candidates.flatten()]
        # concat next tokens
        Y = torch.cat((Y, next_tokens), axis = 1)

    # replace everything after eos token with pad token
    pad_after_eos_mask = Y == end_symbol
    for row in pad_after_eos_mask:
        first_true_found = False
        for i, val in enumerate(row):
            if val and not first_true_found:
                row[i] = False
                first_true_found = True
            elif first_true_found:
                row[i] = True
    Y = Y.masked_fill(pad_after_eos_mask, pad_symbol)

    # length normalization
    ## calculate true length of beam sequences 
    lengths = torch.zeros(batch_size, beam_width, device=device).fill_(max_len) - pad_after_eos_mask.sum(dim=-1).reshape(batch_size, beam_width)
    # normalize
    probabilities = top_probabilities / lengths**length_norm_exp
    _, top_prob_idx = probabilities.sort(dim=-1, descending=True)
    top_prob_idx += torch.arange(batch_size, device=device).unsqueeze(-1) * beam_width
    # take best according to new normalized probs
    Y = Y[top_prob_idx.flatten()]
    probabilities = probabilities.take(top_prob_idx)
    Y = Y.reshape(batch_size, beam_width, -1) # (batch_size, beam_width, max_len)

    if only_best:
        # return first beam of every batch (thats the one with highest probability)
        return Y[:, 0, :]
    return Y, probabilities


def evaluate_beam_search(
    tokenizer,
    model: torch.nn.Module,
    test_dataloader,
    beam_width: int,
    start_symbol,
    pad_symbol,
    end_symbol,
    device
):
    model.eval()

    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tokenizer.decode_batch(tgt.tolist())

        predicted_seqs = beam_search(
            model=model,
            X=src,
            beam_width=beam_width,
            start_symbol=start_symbol,
            pad_symbol=pad_symbol,
            end_symbol=end_symbol,
            device=device,
            max_len=src.shape[1] + 5,
            only_best=True
        )
        predicted_seqs = tokenizer.decode_batch(predicted_seqs.tolist())

    return sacre_bleu_score(predicted_seqs, tgt)


def translate(
    tokenizer,
    model: torch.nn.Module,
    src_sentence: str,
    beam_width: int,
    start_symbol,
    pad_symbol,
    end_symbol,
    device,
    max_len
):
    src = torch.tensor(tokenizer.encode(src_sentence.rstrip("\n")).ids).unsqueeze(dim=0)
    src = src.to(device)
    tgt_tokens = beam_search(
        model=model,
        X=src,
        beam_width=beam_width,
        start_symbol=start_symbol,
        pad_symbol=pad_symbol,
        end_symbol=end_symbol,
        device=device,
        max_len=max_len,
        only_best=True
    )[0]
    return tokenizer.decode(tgt_tokens.tolist())

def greedy_decode(model, src, max_len, start_symbol, pad_symbol, end_symbol, device):
    src = src.to(device)
    src_padding_mask = src == pad_symbol
    memory = model.encode(src, src_padding_mask)
    memory = memory.to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    tgt_padding_mask = ys == pad_symbol
    for _ in range(max_len-1):
        out = model.decode(ys, memory, tgt_padding_mask, src_padding_mask)
        prob = model.unembedding(out[:, -1:, :])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        tgt_padding_mask = ys == pad_symbol
        if next_word == end_symbol:
            break
    return ys

def translate_greedy(tokenizer, model: torch.nn.Module, src_sentence: str, start_symbol, pad_symbol, end_symbol):
    model.eval()
    src = torch.tensor(tokenizer.encode(src_sentence.rstrip("\n")).ids).unsqueeze(dim=0)
    num_tokens = src.shape[1]
    tgt_tokens = greedy_decode(
        model=model,
        src=src,
        max_len=num_tokens + 5,
        start_symbol=start_symbol,
        pad_symbol=pad_symbol,
        end_symbol=end_symbol
    ).flatten()
    return tokenizer.decode(tgt_tokens.tolist())
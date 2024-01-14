import json
import torch
from torchmetrics.functional.text import sacre_bleu_score

with open("params.json", "r") as fp:
    params = json.load(fp)

BOS_IDX = params["BOS_IDX"]
EOS_IDX = params["EOS_IDX"]
PAD_IDX = params["PAD_IDX"]


@torch.no_grad()
def beam_search_batch(
    model,
    X,
    beam_width,
    device,
    max_len = 20,
    only_best: bool = False,
    length_norm_exp: float = 0.6
):
    model.eval()
    batch_size = X.shape[0]
    Y = torch.ones(batch_size, 1, device=device).fill_(BOS_IDX).type(torch.long) # (batch_size, 1)
    src_padding_mask = X == PAD_IDX
    tgt_padding_mask = Y == PAD_IDX
    enc = model.encode(X, src_padding_mask)
    # get decoding for last last token in Y
    dec = model.decode(Y, enc, tgt_padding_mask, src_padding_mask)[:, -1, :] # (batch_size, d_model)
    # get logits for next predicted token
    logits = model.unembedding(dec) # (batch_size, tgt_vocab_size)
    vocabulary_size = logits.shape[-1]
    # search for highest beam_width probabilites within each batch
    next_probabilities, next_tokens = logits.log_softmax(-1).topk(k = beam_width, axis = -1) # (batch_size, beam_width)
    # make form (sample_1, sample_1, ...., sample_n)
    Y = Y.repeat((beam_width, 1)) # (batch_size*beam_width, 1)
    Y = torch.cat((Y, next_tokens.flatten().unsqueeze(dim=1)), dim=-1) # (batch_size*beam_width, 2)
    next_probabilities = next_probabilities.flatten().unsqueeze(1).repeat(1, vocabulary_size) # (batch_size*beam_width, tgt_vocab_size)
    X = X.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=1) # (batch_size*beam_width, src_seq_len)
    src_padding_mask = X == PAD_IDX
    enc = model.encode(X, src_padding_mask) # (batch_size*beam_width, src_seq_len, d_model)
    for _ in range(max_len - 1):
        tgt_padding_mask = Y == PAD_IDX
        dec = model.decode(Y, enc, tgt_padding_mask, src_padding_mask)[:, -1, :] # (batch_size*beam_width, d_model)
        next_logits = model.unembedding(dec) # (batch_size*beam_width, tgt_vocab_size)
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
    pad_after_eos_mask = Y == EOS_IDX
    for row in pad_after_eos_mask:
        first_true_found = False
        for i, val in enumerate(row):
            if val and not first_true_found:
                row[i] = False
                first_true_found = True
            elif first_true_found:
                row[i] = True
    Y = Y.masked_fill(pad_after_eos_mask, PAD_IDX)

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

def get_bleu_score_batch(
    tokenizer,
    model: torch.nn.Module,
    test_dataloader,
    beam_width: int,
    device,
    return_preds: bool = False
):
    preds_all = []
    tgt_all = []
    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tokenizer.decode_batch(tgt.tolist())
        tgt_all.extend(tgt)

        preds = beam_search_batch(
            model=model,
            X=src,
            beam_width=beam_width,
            device=device,
            max_len=src.shape[1] + 5,
            only_best=True
        )
        preds = tokenizer.decode_batch(preds.tolist())
        preds_all.extend(preds)
    if return_preds:
        return sacre_bleu_score(preds_all, [[tgt_item] for tgt_item in tgt_all]), preds_all
    return sacre_bleu_score(preds_all, [[tgt_item] for tgt_item in tgt_all])


@torch.no_grad()
def beam_search_single(model, src, beam_width: int = 3, length_norm_exp: float = 0.7):
    model.eval()
    src_padding_mask = src == PAD_IDX
    memory = model.encode(src, src_padding_mask)
    start_token = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long)

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
    tokenizer,
    model: torch.nn.Module,
    src_sentence: str,
    beam_width: int,
):
    src = torch.tensor(tokenizer.encode(src_sentence).ids).unsqueeze(dim=0)
    tgt_tokens = beam_search_single(
        model=model,
        src=src,
        beam_width=beam_width
    )
    return tokenizer.decode(tgt_tokens.tolist())

def get_bleu_score_single(
    tokenizer,
    model: torch.nn.Module,
    test_data,
    beam_width: int,
    return_preds: bool = False
):
    preds_all = []
    for src in test_data[0]:
        pred = translate(
            tokenizer=tokenizer,
            model=model,
            src=src,
            beam_width=beam_width,
        )
        preds_all.append(pred)
    if return_preds:
        return sacre_bleu_score(preds_all, [[tgt_item] for tgt_item in test_data[1]]), preds_all
    return sacre_bleu_score(preds_all, [[tgt_item] for tgt_item in test_data[1]])

def greedy_decode(model, src, max_len):
    model.eval()
    src_padding_mask = src == PAD_IDX
    memory = model.encode(src, src_padding_mask)
    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long)
    tgt_padding_mask = ys == PAD_IDX
    for _ in range(max_len-1):
        out = model.decode(ys, memory, tgt_padding_mask, src_padding_mask)
        prob = model.unembedding(out[:, -1:, :])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        tgt_padding_mask = ys == PAD_IDX
        if next_word == EOS_IDX:
            break
    return ys.flatten()

def translate_greedy(tokenizer, model: torch.nn.Module, src_sentence: str):
    src = torch.tensor(tokenizer.encode(src_sentence.rstrip("\n")).ids).unsqueeze(dim=0)
    tgt_tokens = greedy_decode(
        model=model,
        src=src,
        max_len=src.shape[1] + 5
    )
    return tokenizer.decode(tgt_tokens.tolist())
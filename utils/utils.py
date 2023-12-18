import torch

@torch.no_grad()
def beam_search(
    model,
    X,
    start_symbol,
    pad_symbol,
    device,
    max_len = 20,
    beam_width = 5
):
    model.eval()
    batch_size = X.shape[0]
    Y = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device) # (batch_size, 1)
    src_padding_mask = X == pad_symbol
    tgt_padding_mask = Y == pad_symbol
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
    src_padding_mask = X == pad_symbol
    enc = model.encode(X, src_padding_mask) # (batch_size*beam_width, src_seq_len, d_model)
    for _ in range(max_len - 1):
        tgt_padding_mask = Y == pad_symbol
        dec = model.decode(Y, enc, tgt_padding_mask, src_padding_mask)[:, -1, :] # (batch_size*beam_width, d_model)
        next_logits = model.unembedding(dec) # (batch_size*beam_width, tgt_vocab_size)
        # adding log probs instead multiplying probs
        next_probabilities += next_logits.log_softmax(-1)
        # search for highest beam_width probabilites within each batch
        top_probabilities, idx = next_probabilities.reshape(batch_size, -1).topk(k = beam_width, axis = -1) # idx (batch_size, beam_width)
        # update to top probabilities
        next_probabilities = top_probabilities.flatten().unsqueeze(-1).repeat(1, vocabulary_size)
        # calculate which tokens correspond to the highest probs
        next_tokens = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1) # (batch_size*beam_width, 1)
        # lookup which current candidates correspond to the highest probs
        best_candidates = idx // vocabulary_size # (batch_size, beam_width)
        # add correct starting index for batches
        best_candidates += torch.arange(batch_size, device = X.device).unsqueeze(-1) * beam_width
        # choose correct beam paths from Y
        Y = Y[best_candidates.flatten()]
        # concat next tokens
        Y = torch.cat((Y, next_tokens), axis = 1)
    # return first beam of every batch (thats the one with highest probability)
    return Y[torch.arange(batch_size, device = device) * beam_width]
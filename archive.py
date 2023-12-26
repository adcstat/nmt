# function to generate output sequence using beam search
def beam_search(model, src, src_mask, max_length, start_symbol, end_symbol, beam_size: int = 3, length_norm_exp: float = 0.7):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    start_token = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # Start the beam with the start token
    beam = [(start_token, 0)]  # List of tuples (sequence, score)

    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1, -1].item() == end_symbol:  # Check for the end of sentence token
                candidates.append((seq, score))
                continue

            tgt_mask = (tfu.generate_square_subsequent_mask(seq.size(0), DEVICE).type(torch.bool)).to(DEVICE)
            out = model.decode(seq, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])  # values before softmax
            topk_probs, topk_indices = torch.topk(prob, beam_size, dim=1)

            # Expand the beam with the top k next words
            for i in range(beam_size):
                next_token = topk_indices[0][i].unsqueeze(0).unsqueeze(0)
                next_score = score + topk_probs[0][i].item()    # adding log probs == multiplying probs
                candidate_seq = torch.cat([seq, next_token], dim=0)
                candidates.append((candidate_seq, next_score))

        # Sort all candidates by score and keep the best ones
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_size]

        if all(seq[-1, -1].item() == end_symbol for seq, _ in beam):
            break

    # Apply length normalization to final sequences
    normalized_beam = [(seq, score / len(seq)**(length_norm_exp)) for seq, score in beam]

    # Choose the sequence with the highest score
    best_sequence, _ = max(beam, key=lambda x: x[1])
    return best_sequence



def beam_search(model, src, src_mask, max_length, start_symbol, end_symbol, beam_size: int = 3, length_norm_exp: float = 0.6):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    batch_size = src.size(1)
    memory = model.encode(src, src_mask)

    # Initialize beams for each item in the batch
    init_token = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    beams = [[(init_token, 0)] for _ in range(batch_size)]
    finished_beams = [False] * batch_size  # Track if beam search is finished for each item

    for _ in range(max_length):
        all_candidates = []
        for i in range(batch_size):
            if finished_beams[i]:
                # Carry forward the existing beam if finished for this item
                all_candidates.append(beams[i])
                continue

            candidates = []
            for seq, score in beams[i]:
                if seq[-1, 0] == end_symbol:
                    candidates.append((seq, score))
                    continue

                seq = seq.to(DEVICE)
                tgt_mask = (tfu.generate_square_subsequent_mask(seq.size(0), DEVICE).type(torch.bool)).to(DEVICE)
                out = model.decode(seq, memory[:, i:i+1], tgt_mask)
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                topk_probs, topk_indices = torch.topk(prob, beam_size, dim=-1)

                for j in range(beam_size):
                    next_token = topk_indices[0][j].unsqueeze(0)
                    next_score = score + topk_probs[0][j].item()
                    candidate_seq = torch.cat([seq, next_token.unsqueeze(0)], dim=0)
                    candidates.append((candidate_seq, next_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            all_candidates.append(candidates[:beam_size])

            if all(seq[-1, 0] == end_symbol for seq, _ in candidates[:beam_size]):
                finished_beams[i] = True

        beams = all_candidates
        if all(finished_beams):
            break

    # Apply length normalization and choose the best sequence for each item in the batch
    best_sequences = []
    for beam in beams:
        normalized_beam = [(seq, score / len(seq)**length_norm_exp) for seq, score in beam]
        best_sequence = max(normalized_beam, key=lambda x: x[1])[0].squeeze()
        best_sequences.append(best_sequence)

    return best_sequences



from torchtext.data.metrics import bleu_score

def evaluate(model, mode: str):
    model.eval()
    all_predicted_seqs = []
    all_target_seqs = []

    iter = Multi30k(split=mode, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    dataloader = DataLoader(iter, batch_size=4, collate_fn=collate_fn)

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            src_mask = tfu.generate_square_subsequent_mask(src.size(0), DEVICE).to(DEVICE)

            predicted_seqs = beam_search(model, src, src_mask, max_length=tgt.size(0),
                                        start_symbol=BOS_IDX, end_symbol=EOS_IDX,
                                        beam_size=1, length_norm_exp=0.6)  # Adjust length_norm_exp as needed

            # Convert predicted and target sequences to the format expected by bleu_score
            for i in range(len(predicted_seqs)):
                predicted_seq = predicted_seqs[i]
                tgt_seq = tgt[:, i]

                # Convert to words or subword tokens as required
                predicted_words = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(predicted_seq.cpu().numpy()))
                target_words = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_seq.cpu().numpy()))

                all_predicted_seqs.append(predicted_words)
                all_target_seqs.append([target_words])

    return bleu_score(all_predicted_seqs, all_target_seqs)


tokenizer_src = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer_tgt = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, tokenizer, index) -> List[str]:
    for data_sample in data_iter:
        yield tokenizer(data_sample['translation'][index])

# Create torchtext's Vocab object
vocab_src = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_src, "de"),
    min_freq=1,
    specials=special_symbols,
    special_first=True
)
vocab_tgt = build_vocab_from_iterator(
    yield_tokens(train_iter, tokenizer_tgt, "en"),
    min_freq=1,
    specials=special_symbols,
    special_first=True
)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
vocab_src.set_default_index(UNK_IDX)
vocab_tgt.set_default_index(UNK_IDX)


# function to add BOS/EOS and create tensor for input sequence indices
def add_bos_eos(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
string_to_indices_src = sequential_transforms(tokenizer_src, vocab_src, add_bos_eos)
string_to_indices_tgt = sequential_transforms(tokenizer_tgt, vocab_tgt, add_bos_eos)

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_batch.append(string_to_indices_src(sample['translation']["de"].rstrip("\n")))
        tgt_batch.append(string_to_indices_tgt(sample['translation']["en"].rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch


def evaluate_beam_search(model, beam_width: int):
    model.eval()
    all_predicted_seqs = []
    all_target_seqs = []

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        predicted_seqs = utils.beam_search(model, src, BOS_IDX, PAD_IDX, EOS_IDX, DEVICE, src.shape[1] + 5, beam_width, True)

        # Convert predicted and target sequences to the format expected by bleu_score
        for i in range(src.shape[0]):
            predicted_seq = predicted_seqs[i]
            tgt_seq = tgt[i]

            # Convert to words or subword tokens as required
            predicted_words = vocab_tgt.lookup_tokens(predicted_seq.tolist())
            predicted_words = [tok for tok in predicted_words if tok not in ["<bos>", "<pad>", "<eos>"]]
            target_words = vocab_tgt.lookup_tokens(tgt_seq.tolist())
            target_words = [tok for tok in target_words if tok not in ["<bos>", "<pad>", "<eos>"]]

            all_predicted_seqs.append(predicted_words)
            all_target_seqs.append([target_words])

    return bleu_score(all_predicted_seqs, all_target_seqs)


def train_epoch(model):
    model.train()
    losses = np.array([])

    for batch_i, (src, tgt) in enumerate(train_dataloader):
        batch_i += 1
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_in == PAD_IDX

        with torch.cuda.amp.autocast():
            logits = transformer(src, tgt_in, src_padding_mask, tgt_padding_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        # Gradient Accumulation
        if batch_i % 4 == 0:
          # Scales loss and performs backward pass using automatic mixed precision
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad(set_to_none=True)

        losses = np.append(losses, np.array([loss.item()]))
        if batch_i % 1000 == 0:
            print(f"accumulated loss so far (batch {batch_i}): ", losses.sum() / batch_i)

        del src
        del tgt
        del src_padding_mask
        del tgt_padding_mask
        torch.cuda.empty_cache()

    return losses / train_dataloader_len

@torch.no_grad()
def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]

        src_padding_mask = src == PAD_IDX
        tgt_padding_mask = tgt_input == PAD_IDX

        logits = transformer(src, tgt_input, src_padding_mask, tgt_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / val_dataloader_len


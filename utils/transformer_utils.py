from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        maxlen: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model) # this equates to 1 / 10000^(2i/d_model) with 2i in torch.arange(0, d_model, 2)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # this is PE_(pos, 2i)
        pos_embedding[:, 1::2] = torch.cos(pos * den) # this is PE_(pos, 2i+1)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.shape[1]]

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout, masked):
        super().__init__()
        self.d_k = d_k
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_v, bias=False)
        self.masked = masked
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask):
        # source_query of shape (batch_size, seq_len_q, d_model)
        # source_key_value of shape (batch_size, seq_len_kv, d_model)
        # source_query_padding_mask of shape (batch_size, seq_len_q)
        # output of shape (batch_size, seq_len_q, d_v)
        q = self.query(source_query) # (batch_size, seq_len_q, d_k)
        k = self.key(source_key_value) # (batch_size, seq_len_kv, d_k)
        # compute attention scores ("affinities")
        attention_weights = q @ k.transpose(-2,-1) * self.d_k**-0.5 # (batch_size, seq_len_q, d_k) @ (batch_size, d_k, seq_len_kv) -> (batch_size, seq_len_q, seq_len_kv)
        # padding mask
        stretched_source_query_padding_mask = source_query_padding_mask.unsqueeze(dim=1).repeat(1, source_key_value.shape[1], 1).squeeze().transpose(-2, -1)
        attention_weights = attention_weights.masked_fill(stretched_source_query_padding_mask, float('-inf'))
        stretched_source_key_value_padding_mask = source_key_value_padding_mask.unsqueeze(dim=1).repeat(1, source_query.shape[1], 1).squeeze()
        attention_weights = attention_weights.masked_fill(stretched_source_key_value_padding_mask, float('-inf'))
        # autoregressive masking only makes sense for source_query == source_key_value
        if self.masked:
            mask = torch.tril(torch.ones(attention_weights.shape[1], attention_weights.shape[1]))
            mask = mask.to(source_query.device)
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf')) # (batch_size, seq_len_q, seq_len_kv)
        attention_weights = F.softmax(attention_weights, dim=-1) # (batch_size, seq_len_q, seq_len_kv)
        # since the rows and pad tokens only contain -inf and therefore nan after softmax we replace with 0 
        attention_weights = attention_weights.masked_fill(attention_weights.isnan(), 0)
        attention_weights = self.dropout(attention_weights)
        # perform the weighted aggregation of the values
        v = self.value(source_key_value) # (batch_size, seq_len_kv, d_v)
        out = attention_weights @ v # (batch_size, seq_len_q, seq_len_kv) @ (batch_size, seq_len_kv, d_v) -> (batch_size, seq_len_q, d_v)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_heads, d_model, dropout, masked):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_k
        self.heads = nn.ModuleList([Attention(d_model, d_k, d_v, dropout, masked) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask):
        out = torch.cat([h(source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask) for h in self.heads], dim=-1) # (batch_size, seq_len_q, n_heads*d_v)
        out = self.dropout(self.proj(out)) # (batch_size, seq_len_q, d_model)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, masked):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = FeedFoward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask):
        ln1 = self.ln1(src)
        sa = src + self.self_attention(
            source_query=ln1,
            source_key_value=ln1,
            source_query_padding_mask=src_padding_mask,
            source_key_value_padding_mask=src_padding_mask
        )
        out = sa + self.ffwd(self.ln2(sa))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=True)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=False)
        self.ffwd = FeedFoward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_padding_mask, memory_padding_mask):
        ln1 = self.ln1(tgt)
        sa = tgt + self.self_attention(
            source_query=ln1,
            source_key_value=ln1,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=tgt_padding_mask
        )
        ca = sa + self.cross_attention(
            source_query=self.ln2(sa),
            source_key_value=self.ln3(memory),
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=memory_padding_mask
        )
        out = ca + self.ffwd(self.ln4(ca))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_encoder_layers,
        n_decoder_layers,
        dropout
    ):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_decoder_layers)])

        self.ln_final = nn.LayerNorm(d_model) # final layer norm before unembedding
        self.unembedding = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        enc = src_emb
        for layer in self.encoder:
            enc = layer(enc, src_padding_mask)

        dec = tgt_emb
        for layer in self.decoder:
            dec = layer(dec, enc, tgt_padding_mask, src_padding_mask)

        decoder_output_norm = self.ln_final(dec)
        logits = self.unembedding(decoder_output_norm)

        return logits
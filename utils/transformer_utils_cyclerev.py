"""
This module only differs from tranformer_utils_plain_vanilla by implementing Admin residual connections layer cycling.
Look up said module for informative docstrings :)
"""
from torch import Tensor
import torch
from torch import nn
import admin_torch
import math

PADDING_IDX = 2

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout,
        maxlen: int = 5000,
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model) # this equates to 1 / 10000^(2i/d_model) with 2i in torch.arange(0, d_model, 2)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # this is PE_(pos, 2i)
        pos_embedding[:, 1::2] = torch.cos(pos * den) # this is PE_(pos, 2i+1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[1]])

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout, masked):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.masked = masked

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x.shape = (batch size, seq_length, d_model)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)
        x = x.permute(0, 2, 1, 3)
        return x # (batch size, n_heads, seq_length, d_k)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask):
        # source_query of shape (batch_size, seq_len_q, d_model)
        # source_key_value of shape (batch_size, seq_len_kv, d_model)
        # source_query_padding_mask of shape (batch_size, seq_len_q)
        # output of shape (batch_size, seq_len_q, d_v)
        q = self.split_heads(self.query(source_query))
        k = self.split_heads(self.key(source_key_value))
        v = self.split_heads(self.value(source_key_value))
        # compute attention scores
        attention_weights_raw = q @ k.transpose(-2,-1) * self.d_k**-0.5  # (batch_size, n_heads, seq_len_q, d_k) @ (batch_size, n_heads, d_k, seq_len_kv) -> (batch_size, n_heads, seq_len_q, seq_len_kv)
        # padding masking
        stretched_source_query_padding_mask = source_query_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_heads, source_key_value.shape[1], 1).transpose(-2, -1)
        attention_weights_raw = attention_weights_raw.masked_fill(stretched_source_query_padding_mask, float('-inf'))
        stretched_source_key_value_padding_mask = source_key_value_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_heads, source_query.shape[1], 1)
        attention_weights_raw = attention_weights_raw.masked_fill(stretched_source_key_value_padding_mask, float('-inf'))
        # causal masking (only applicable for source_query == source_key_value)
        if self.masked:
            mask = torch.tril(torch.ones(source_query.shape[1], source_query.shape[1], device=source_query.device))
            attention_weights_raw = attention_weights_raw.masked_fill(mask == 0, float('-inf')) # (batch_size, n_heads, seq_len_q, seq_len_kv)
        # soft max rows of attention_weights_raw
        attention_weights_inter = attention_weights_raw.softmax(-1) # (batch_size, n_heads, seq_len_q, seq_len_kv)
        # since the rows of pad tokens only contain -inf and therefore nan after softmax we replace with 0
        attention_weights = attention_weights_inter.masked_fill(attention_weights_inter.isnan(), 0)
        # perform the weighted aggregation of the values
        attention = attention_weights @ v # (batch_size, n_heads, seq_len_q, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, d_v) -> (batch_size, n_heads, seq_len_q, d_v)
        # flatten along heads
        attention = attention.permute(0, 2, 1, 3).flatten(-2) # (batch_size, seq_len_q, d_model)
        # projection
        attention = self.dropout(self.proj(attention)) # (batch_size, seq_len_q, d_model)
        return attention, attention_weights_inter

class FeedForward(nn.Module):
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
    def __init__(self, n_heads, d_model, d_ff, dropout, n_layers, masked):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.residual_attn = admin_torch.as_module(2 * n_layers)
        self.residual_ffn = admin_torch.as_module(2 * n_layers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask):
        attention, attention_weights = self.self_attention(
            source_query=src,
            source_key_value=src,
            source_query_padding_mask=src_padding_mask,
            source_key_value_padding_mask=src_padding_mask
        )
        sa_norm = self.ln1(self.residual_attn(attention, src))
        out = self.ln2(self.residual_ffn(self.ffwd(sa_norm), sa_norm))
        return out, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, n_layers):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=True)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=False)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.residual_sattn = admin_torch.as_module(3 * n_layers)
        self.residual_cattn = admin_torch.as_module(3 * n_layers)
        self.residual_ffn = admin_torch.as_module(3 * n_layers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_padding_mask, memory_padding_mask):
        sa, sa_weights = self.self_attention(
            source_query=tgt,
            source_key_value=tgt,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=tgt_padding_mask,
        )
        sa_norm = self.ln1(self.residual_sattn(sa, tgt))
        ca, ca_weights = self.cross_attention(
            source_query=sa_norm,
            source_key_value=memory,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=memory_padding_mask,
        )
        ca_norm = self.ln2(self.residual_cattn(ca, sa_norm))
        out = self.ln3(self.residual_ffn(self.ffwd(ca_norm), ca_norm))
        return out, sa_weights, ca_weights

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_encoder_layers,
        n_decoder_layers,
        dropout
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PADDING_IDX)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, n_encoder_layers, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout, n_decoder_layers) for _ in range(n_decoder_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def encode(self, src: Tensor, src_padding_mask: Tensor):
        enc = self.positional_encoding(self.tok_emb(src.long()) * self.d_model**0.5)
        for layer in self.encoder:
            enc, _ = layer(enc, src_padding_mask)
        # cycle rev:
        for layer in reversed(self.encoder):
            enc, _ = layer(enc, src_padding_mask)
        return enc

    def decode(self, tgt, enc, tgt_padding_mask, src_padding_mask):
        dec = self.positional_encoding(self.tok_emb(tgt.long()) * self.d_model**0.5)
        for layer in self.decoder:
            dec, _, _ = layer(dec, enc, tgt_padding_mask, src_padding_mask)
        # cycle rev
        for layer in reversed(self.decoder):
            dec, _, _ = layer(dec, enc, tgt_padding_mask, src_padding_mask)
        return dec

    def unembedding(self, dec):
        logits = dec @ self.tok_emb.weight.T
        return logits

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor
    ):
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(tgt, enc, tgt_padding_mask, src_padding_mask)
        logits = self.unembedding(dec)

        return logits

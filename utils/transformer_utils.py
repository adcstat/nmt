from torch import Tensor
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout, masked):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
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
        stretched_source_query_padding_mask = source_query_padding_mask.unsqueeze(dim=1).repeat(1, source_key_value.shape[1], 1).transpose(-2, -1)
        attention_weights = attention_weights.masked_fill(stretched_source_query_padding_mask, float('-inf'))
        stretched_source_key_value_padding_mask = source_key_value_padding_mask.unsqueeze(dim=1).repeat(1, source_query.shape[1], 1)
        attention_weights = attention_weights.masked_fill(stretched_source_key_value_padding_mask, float('-inf'))
        # autoregressive masking only makes sense for source_query == source_key_value
        if self.masked:
            mask = torch.tril(torch.ones(attention_weights.shape[1], attention_weights.shape[1], device=source_query.device))
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf')) # (batch_size, seq_len_q, seq_len_kv)
        attention_weights = attention_weights.softmax(-1) # (batch_size, seq_len_q, seq_len_kv)
        # since the rows of pad tokens only contain -inf and therefore nan after softmax we replace with 0 
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
        self.proj = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask):
        out = torch.cat([h(source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask) for h in self.heads], dim=-1) # (batch_size, seq_len_q, n_heads*d_v)
        out = self.dropout(self.proj(out)) # (batch_size, seq_len_q, d_model)
        return out


class SwiGLUFeedFoward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.swish = nn.SiLU()
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W_2(self.swish(self.W(x)) * self.V(x)))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, masked):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = SwiGLUFeedFoward(d_model, d_ff, dropout)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

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
        self.ffwd = SwiGLUFeedFoward(d_model, d_ff, dropout)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ln3 = RMSNorm(d_model)
        self.ln4 = RMSNorm(d_model)

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
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_decoder_layers)])

        self.ln_final = RMSNorm(d_model) # final layer norm before unembedding

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_padding_mask: Tensor):
        # no need to multiply by sqrt(d_model), since it gets feed into layer norm immediately
        enc = self.tok_emb(src.long())
        for layer in self.encoder:
            enc = layer(enc, src_padding_mask)
        return enc

    def decode(self, tgt, enc, tgt_padding_mask, src_padding_mask):
        dec = self.tok_emb(tgt.long())
        for layer in self.decoder:
            dec = layer(dec, enc, tgt_padding_mask, src_padding_mask)
        dec = self.ln_final(dec)
        return dec

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor
    ):
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(tgt, enc, tgt_padding_mask, src_padding_mask)

        logits = dec.reshape(dec.shape[0] * dec.shape[1], -1) @ self.tok_emb.weight.data.T
        logits = logits.reshape(dec.shape[0], dec.shape[1], -1)

        return logits


class RMSNorm(nn.Module):
    def __init__(self, size: int, p: float = None):
        super().__init__()
        self.size = size
        self.p = p
        self.gamma = torch.nn.Parameter(torch.ones(size))

    def forward(self, x):
        if x.shape[-1] != self.size:
            raise ValueError("Last dimension of tensor x must have same size that RMSNorm was initialized with")
        if self.p:
            k = int(self.p * self.size)
            x_red = x[..., :k]
            rms =  ((x_red**2).sum(-1)/self.size)**0.5
        else:
            rms = ((x**2).sum(-1)/self.size)**0.5
        rms = rms.unsqueeze(-1)
        return self.gamma * x / rms

    def __repr__(self):
        return f"RMSNorm(size={self.size}, p={self.p})"
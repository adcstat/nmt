from torch import Tensor
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        maxlen: int = 5000
    ):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model) # this equates to 1 / 10000^(2i/d_model) with 2i in torch.arange(0, d_model, 2)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # this is PE_(pos, 2i)
        pos_embedding[:, 1::2] = torch.cos(pos * den) # this is PE_(pos, 2i+1)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.shape[1]]


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout, masked):
        super().__init__()
        self.d_k = d_k
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_v, bias=False)
        self.masked = masked
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask, prev_i, layer_ind):
        # source_query of shape (batch_size, seq_len_q, d_model)
        # source_key_value of shape (batch_size, seq_len_kv, d_model)
        # source_query_padding_mask of shape (batch_size, seq_len_q)
        # output of shape (batch_size, seq_len_q, d_v)
        q = self.query(source_query) # (batch_size, seq_len_q, d_k)
        k = self.key(source_key_value) # (batch_size, seq_len_kv, d_k)
        # compute attention scores ("affinities")
        attention_weights_raw = q @ k.transpose(-2,-1) * self.d_k**-0.5 # (batch_size, seq_len_q, d_k) @ (batch_size, d_k, seq_len_kv) -> (batch_size, seq_len_q, seq_len_kv)
        # padding mask
        stretched_source_query_padding_mask = source_query_padding_mask.unsqueeze(dim=1).repeat(1, source_key_value.shape[1], 1).transpose(-2, -1)
        attention_weights_raw = attention_weights_raw.masked_fill(stretched_source_query_padding_mask, float('-inf'))
        stretched_source_key_value_padding_mask = source_key_value_padding_mask.unsqueeze(dim=1).repeat(1, source_query.shape[1], 1)
        attention_weights_raw = attention_weights_raw.masked_fill(stretched_source_key_value_padding_mask, float('-inf'))
        # autoregressive masking only makes sense for source_query == source_key_value
        if self.masked:
            mask = torch.tril(torch.ones(attention_weights_raw.shape[1], attention_weights_raw.shape[1], device=source_query.device))
            attention_weights_raw = attention_weights_raw.masked_fill(mask == 0, float('-inf')) # (batch_size, seq_len_q, seq_len_kv)
        # Realformer residual connections using exp weighted avg with beta=0.5
        attention_weights_raw = (attention_weights_raw + prev_i) / 2
        # bias correction
        attention_weights_cor = attention_weights_raw / (1-0.5**layer_ind)
        attention_weights = attention_weights_cor.softmax(-1) # (batch_size, seq_len_q, seq_len_kv)
        # since the rows of pad tokens only contain -inf and therefore nan after softmax we replace with 0 SwiGLUFeedForward
        attention_weights = attention_weights.masked_fill(attention_weights.isnan(), 0)
        attention_weights = self.dropout(attention_weights)
        # perform the weighted aggregation of the values
        v = self.value(source_key_value) # (batch_size, seq_len_kv, d_v)
        attention = attention_weights @ v # (batch_size, seq_len_q, seq_len_kv) @ (batch_size, seq_len_kv, d_v) -> (batch_size, seq_len_q, d_v)
        return attention, attention_weights_raw


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_heads, d_model, dropout, masked):
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_k
        self.heads = nn.ModuleList([Attention(d_model, d_k, d_v, dropout, masked) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask, prev, layer_ind):
        att_outs = [h(source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask, prev[i], layer_ind) for i, h in enumerate(self.heads)]
        attention = torch.cat([out[0] for out in att_outs], dim=-1) # (batch_size, seq_len_q, n_heads*d_v)
        attention = self.dropout(self.proj(attention)) # (batch_size, seq_len_q, d_model)
        attention_weights_raw = [out[1] for out in att_outs]
        return attention, attention_weights_raw


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.swish = nn.SiLU()
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W_2(self.swish(self.W(x)) * self.V(x)))

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
    def __init__(self, n_heads, d_model, d_ff, dropout, masked):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, src, src_padding_mask, prev, layer_ind):
        ln1 = self.ln1(src)
        attention, attention_weights_raw = self.self_attention(
            source_query=ln1,
            source_key_value=ln1,
            source_query_padding_mask=src_padding_mask,
            source_key_value_padding_mask=src_padding_mask,
            prev=prev,
            layer_ind=layer_ind
        )
        attention += src
        out = attention + self.ffwd(self.ln2(attention))
        return out, attention_weights_raw


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=True)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=False)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ln3 = RMSNorm(d_model)
        self.ln4 = RMSNorm(d_model)

    def forward(self, tgt, memory, tgt_padding_mask, memory_padding_mask, prev_sa, prev_ca, layer_ind):
        ln1 = self.ln1(tgt)
        sa, sa_weights_raw = self.self_attention(
            source_query=ln1,
            source_key_value=ln1,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=tgt_padding_mask,
            prev=prev_sa,
            layer_ind=layer_ind
        )
        sa += tgt
        ca, ca_weights_raw = self.cross_attention(
            source_query=self.ln2(sa),
            source_key_value=self.ln3(memory),
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=memory_padding_mask,
            prev=prev_ca,
            layer_ind=layer_ind
        )
        ca += sa
        out = ca + self.ffwd(self.ln4(ca))
        return out, sa_weights_raw, ca_weights_raw


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
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_decoder_layers)])

        self.ln_final = nn.LayerNorm(d_model) # final layer norm before unembedding
        self.unembedding = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_padding_mask: Tensor):
        enc = self.positional_encoding(self.tok_emb(src.long()) * self.d_model**0.5)
        prev = [torch.zeros(src.shape[0], src.shape[1], src.shape[1], device=src.device)] * self.n_heads
        for layer_ind, layer in enumerate(self.encoder):
            enc, prev = layer(enc, src_padding_mask, prev, layer_ind+1)
        return enc

    def decode(self, tgt, enc, tgt_padding_mask, src_padding_mask):
        dec = self.positional_encoding(self.tok_emb(tgt.long()) * self.d_model**0.5)
        prev_sa = [torch.zeros(tgt.shape[0], tgt.shape[1], tgt.shape[1], device=tgt.device)] * self.n_heads
        prev_ca = [torch.zeros(tgt.shape[0], tgt.shape[1], enc.shape[1], device=tgt.device)] * self.n_heads
        for layer_ind, layer in enumerate(self.decoder):
            dec, prev_sa, prev_ca = layer(dec, enc, tgt_padding_mask, src_padding_mask, prev_sa, prev_ca, layer_ind+1)
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
        logits = self.unembedding(dec)

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
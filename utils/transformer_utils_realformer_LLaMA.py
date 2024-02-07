"""
This module only differs from tranformer_utils_plain_vanilla by implementing Admin residual connections and Realformer attention residuals
and some ideas from LLaMA:
Rotary positional Encoding
SwiGLU MLP
Look up said module for informative docstrings :)
"""
import math
from torch import Tensor
import torch
from torch import nn
import admin_torch

PADDING_IDX = 2

class RoPE(nn.Module):
    """
    Implements the Rotational Positional Encoding (RoPE) module.

    Args:
        dim (int): The dimensionality of the input feature space.
        max_len (int): The maximum length of the input sequences. Defaults to 5000.

    Attributes:
        dim (int): The dimensionality of the input feature space.
        max_len (int): The maximum length of the input sequences. Defaults to 5000.
        Theta (Tensor): A tensor containing the rotational angles for positional encoding.
        sin (Tensor): The sine component of the positional encodings.
        cos (Tensor): The cosine component of the positional encodings.
    """
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.Theta = torch.exp(- torch.arange(dim) * math.log(10000) / dim).unsqueeze(-1).repeat(1, 2).flatten()
        self.range_tensor = torch.arange(1, max_len + 1).unsqueeze(1).expand(-1, 2*dim)
        self.register_buffer('sin', torch.sin(self.range_tensor * self.Theta))
        self.register_buffer('cos', torch.cos(self.range_tensor * self.Theta))

    def reshape_elements(self, x: Tensor):
        """
        Reshapes the input tensor to facilitate rotational encoding.

        This method splits the last dimension of the input tensor into pairs, negates the
        even-indexed elements, and then recombines the elements to their original ordering.

        Args:
            x (Tensor): The input tensor with shape (..., dim).

        Returns:
            Tensor: The reshaped tensor with the same shape as the input.
        """
        # Reshape the tensor to split the last dimension into pairs
        reshaped = x.clone().reshape(*x.shape[:-1], -1, 2)
        # Negate the even-indexed elements (which are now at index 1 of the innermost dimension due to reshape)
        reshaped[..., 1] = -reshaped[..., 1]
        # Swap the pairs back to the original order and flatten the last two dimensions
        modified = reshaped.flip(-1).reshape(x.shape)
        return modified

    def forward(self, x: Tensor):
        """
        Applies rotational positional encoding to the input tensor.

        Args:
            x (Tensor): The input tensor with shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: The positionally encoded tensor with the same shape as the input.
        """
        x_reshaped = self.reshape_elements(x)
        sequence_length = x.shape[-2]
        Rx = x * self.cos[:sequence_length] + x_reshaped * self.sin[:sequence_length]
        return Rx

    def __repr__(self):
        return f"RoPE(dim={self.dim})"

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout, masked):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.RoPE = RoPE(self.d_k // 2)
        self.masked = masked

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x.shape = (batch size, seq_length, d_model)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)
        x = x.permute(0, 2, 1, 3)
        return x # (batch size, n_heads, seq_length, d_k)

    def forward(self, source_query, source_key_value, source_query_padding_mask, source_key_value_padding_mask, prev):
        # source_query of shape (batch_size, seq_len_q, d_model)
        # source_key_value of shape (batch_size, seq_len_kv, d_model)
        # source_query_padding_mask of shape (batch_size, seq_len_q)
        # output of shape (batch_size, seq_len_q, d_v)
        q = self.split_heads(self.query(source_query))
        k = self.split_heads(self.key(source_key_value))
        v = self.split_heads(self.value(source_key_value))
        # rotate according to RoPE
        q = self.RoPE(q)
        k = self.RoPE(k)
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
        # realformer running mean residuals
        ## init
        if prev is None:
            prev_new = attention_weights_raw.unsqueeze(0)
        else:
            prev_new = torch.cat((prev, attention_weights_raw.unsqueeze(0)), 0)
             # (n_layers, batch_size, n_heads, seq_len_q, seq_len_kv)
        attention_weights_avg = prev_new.mean(0)
        # soft max rows of attention_weights_avg
        attention_weights_inter = attention_weights_avg.softmax(-1) # (batch_size, n_heads, seq_len_q, seq_len_kv)
        # since the rows of pad tokens only contain -inf and therefore nan after softmax we replace with 0
        attention_weights = attention_weights_inter.masked_fill(attention_weights_inter.isnan(), 0)
        # perform the weighted aggregation of the values
        attention = attention_weights @ v # (batch_size, n_heads, seq_len_q, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, d_v) -> (batch_size, n_heads, seq_len_q, d_v)
        # flatten along heads
        attention = attention.permute(0, 2, 1, 3).flatten(-2) # (batch_size, seq_len_q, d_model)
        # projection
        attention = self.dropout(self.proj(attention)) # (batch_size, seq_len_q, d_model)
        return attention, prev_new

class SwiGLUFeedForward(nn.Module):
    """
    Implements a feedforward neural network module with SwiGLU activation.

    Args:
        d_model (int): The dimensionality of the input and output feature space.
        d_ff (int): The dimensionality of the hidden layer.
        dropout (float): The dropout rate for regularization. Defaults to 0.

    Attributes:
        W (nn.Linear): The first linear transformation applied to the input.
        swish (nn.SiLU): The Swish activation function, applied to the output of `W`.
        V (nn.Linear): A linear transformation applied to the original input, serving as a gate when
                       multiplied with the Swish-activated output.
        W_2 (nn.Linear): The second linear transformation, applied to the gated output.
        dropout (nn.Dropout): Dropout layer applied to the final output for regularization.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.swish = nn.SiLU()
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W_2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Defines the forward pass of the SwiGLUFeedForward module.

        Args:
            x (Tensor): The input tensor with shape `(batch_size, seq_len, d_model)`.

        Returns:
            Tensor: The output tensor with the same shape as the input.
        """
        return self.dropout(self.W_2(self.swish(self.W(x)) * self.V(x)))

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, n_layers, masked):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.residual_attn = admin_torch.as_module(2 * n_layers)
        self.residual_ffn = admin_torch.as_module(2 * n_layers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, src, src_padding_mask, prev):
        attention, attention_weights = self.self_attention(
            source_query=src,
            source_key_value=src,
            source_query_padding_mask=src_padding_mask,
            source_key_value_padding_mask=src_padding_mask,
            prev=prev
        )
        sa_norm = self.ln1(self.residual_attn(attention, src))
        out = self.ln2(self.residual_ffn(self.ffwd(sa_norm), sa_norm))
        return out, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, n_layers):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=True)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=False)
        self.ffwd = SwiGLUFeedForward(d_model, d_ff, dropout)
        self.residual_sattn = admin_torch.as_module(3 * n_layers)
        self.residual_cattn = admin_torch.as_module(3 * n_layers)
        self.residual_ffn = admin_torch.as_module(3 * n_layers)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_padding_mask, memory_padding_mask, prev_sa, prev_ca):
        sa, sa_weights = self.self_attention(
            source_query=tgt,
            source_key_value=tgt,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=tgt_padding_mask,
            prev=prev_sa
        )
        sa_norm = self.ln1(self.residual_sattn(sa, tgt))
        ca, ca_weights = self.cross_attention(
            source_query=sa_norm,
            source_key_value=memory,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=memory_padding_mask,
            prev=prev_ca
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

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, n_encoder_layers, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout, n_decoder_layers) for _ in range(n_decoder_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    @torch.no_grad()
    def get_attention_weights(self, src, tgt, src_padding_mask, tgt_padding_mask):
        enc, prev = self.encode(src, src_padding_mask, True)
        _, prev_sa, prev_ca = self.decode(tgt, enc, tgt_padding_mask, src_padding_mask, True)
        enc_att_weights_all = prev.permute(0, 2, 1, 3, 4).softmax(-1)
        dec_self_att_weights_all = prev_sa.permute(0, 2, 1, 3, 4).softmax(-1)
        enc_dec_weights_all = prev_ca.permute(0, 2, 1, 3, 4).softmax(-1)
        return enc_att_weights_all, dec_self_att_weights_all, enc_dec_weights_all

    def encode(self, src: Tensor, src_padding_mask: Tensor, return_prev: bool = False):
        enc = self.tok_emb(src.long()) * self.d_model**0.5
        prev = None
        for layer in self.encoder:
            enc, prev = layer(enc, src_padding_mask, prev)
        if return_prev:
            return enc, prev
        return enc

    def decode(self, tgt, enc, tgt_padding_mask, src_padding_mask, return_prev: bool = False):
        dec = self.tok_emb(tgt.long()) * self.d_model**0.5
        prev_sa, prev_ca = None, None
        for layer in self.decoder:
            dec, prev_sa, prev_ca = layer(dec, enc, tgt_padding_mask, src_padding_mask, prev_sa, prev_ca)
        if return_prev:
            return dec, prev_sa, prev_ca
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

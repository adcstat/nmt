"""
This module implements a plain vanilla Transformer model
"""
import math
from typing import Tuple
from torch import Tensor
import torch
from torch import nn

PADDING_IDX = 2

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to token embeddings to retain the sequence order information.

    Args:
        d_model (int): The dimensionality of the input embeddings.
        dropout (float): The dropout rate to be applied to the positional encodings.
        maxlen (int, optional): The maximum length of the input sequences. Defaults to 5000.

    Attributes:
        dropout (nn.Dropout): Dropout layer to be applied to the positional encodings.
        pos_embedding (Tensor): Precomputed positional encoding matrix.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float,
        maxlen: int = 5000,
    ):
        super().__init__()
        # this equates to 1 / 10000^(2i/d_model) with 2i in torch.arange(0, d_model, 2)
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den) # this is PE_(pos, 2i)
        pos_embedding[:, 1::2] = torch.cos(pos * den) # this is PE_(pos, 2i+1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """
        Adds positional encoding to token embeddings.

        Args:
            token_embedding (Tensor): The token embeddings.

        Returns:
            Tensor: The token embeddings with added positional encodings.
        """
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.shape[1]])

class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism.
    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Total dimension of the model.
        dropout (float): Dropout rate.
        masked (bool): If True, applies masking to prevent positions from attending to subsequent positions.

    Attributes:
        query, key, value (nn.Linear): Linear projections for query, key, and value vectors.
        proj (nn.Linear): Linear projection of concatenated head outputs.
        dropout (nn.Dropout): Dropout layer applied to the output of the attention.
    """
    def __init__(self, n_heads: int, d_model: int, dropout: float, masked: bool):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.masked = masked

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: Tensor) -> Tensor:
        """
        Splits the last dimension of the input tensor into (n_heads, depth) and transposes the result for multi-head attention.

        This operation allows the model to process data with multiple attention heads simultaneously, increasing the model's
        capacity to focus on different parts of the sequence independently.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            Tensor: Reorganized tensor of shape (batch_size, n_heads, seq_length, depth), where depth = d_model // n_heads.
            This rearrangement allows each head to attend independently on the sequence.
        """
        # x.shape = (batch size, seq_length, d_model)
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.d_k)
        x = x.permute(0, 2, 1, 3)
        return x # (batch size, n_heads, seq_length, d_k)

    def forward(
        self,
        source_query: Tensor,
        source_key_value: Tensor,
        source_query_padding_mask: Tensor,
        source_key_value_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the multi-head attention.

        Args:
            source_query (Tensor): Query sequences.
            source_key_value (Tensor): Key and Value sequences.
            source_query_padding_mask (Tensor): Mask for query sequences to ignore padding tokens.
            source_key_value_padding_mask (Tensor): Mask for key/value sequences to ignore padding tokens.

        Returns:
            Tuple[Tensor, Tensor]: The output after attention and the attention weights.
        """
        # source_query of shape (batch_size, seq_len_q, d_model)
        # source_key_value of shape (batch_size, seq_len_kv, d_model)
        # source_query_padding_mask of shape (batch_size, seq_len_q)
        # output of shape (batch_size, seq_len_q, d_v)
        q = self.split_heads(self.query(source_query))
        k = self.split_heads(self.key(source_key_value))
        v = self.split_heads(self.value(source_key_value))
        # compute attention scores
        # (batch_size, n_heads, seq_len_q, d_k) @ (batch_size, n_heads, d_k, seq_len_kv) -> (batch_size, n_heads, seq_len_q, seq_len_kv)
        attention_weights_raw = q @ k.transpose(-2,-1) * self.d_k**-0.5
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
    """
    Implements the feed-forward layer in the Transformer architecture.

    This layer applies two linear transformations with a ReLU activation in between.

    Args:
        d_model (int): Dimensionality of the input tensor.
        d_ff (int): Dimensionality of the hidden layer.
        dropout (float): Dropout rate.

    Attributes:
        net (nn.Sequential): The feed-forward network.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the feed-forward network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.net(x)

class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer encoder.

    Each layer consists of a multi-head self-attention mechanism followed by a position-wise fully connected feed-forward network.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer.
        dropout (float): Dropout rate.
        masked (bool): If True, applies masking (unused in encoder, relevant for decoder).

    Attributes:
        self_attention (MultiHeadAttention): The multi-head self-attention mechanism.
        ffwd (FeedForward): The feed-forward network.
        ln1, ln2 (nn.LayerNorm): Layer normalization.
    """
    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float, masked: bool):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=masked)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes input through one encoder layer.

        Args:
            src (Tensor): Input tensor to the encoder layer.
            src_padding_mask (Tensor): Mask for the input tensor to ignore padding tokens.

        Returns:
            Tuple[Tensor, Tensor]: The output tensor and self-attention weights.
        """
        attention, attention_weights = self.self_attention(
            source_query=src,
            source_key_value=src,
            source_query_padding_mask=src_padding_mask,
            source_key_value_padding_mask=src_padding_mask
        )
        sa_norm = self.ln1(attention + src)
        out = self.ln2(self.ffwd(sa_norm) + sa_norm)
        return out, attention_weights

class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer decoder.

    Each layer consists of three sub-layers: a multi-head self-attention mechanism, a multi-head cross-attention
    mechanism with the encoder output, and a position-wise fully connected feed-forward network.

    Args:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer.
        dropout (float): Dropout rate.

    Attributes:
        self_attention (MultiHeadAttention): The multi-head self-attention mechanism for the decoder input.
        cross_attention (MultiHeadAttention): The multi-head attention mechanism that attends to the encoder's output.
        ffwd (FeedForward): The feed-forward network.
        ln1, ln2, ln3 (nn.LayerNorm): Layer normalization.
    """
    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=True)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout, masked=False)
        self.ffwd = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_padding_mask: Tensor,
        memory_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Processes input through one decoder layer.

        Args:
            tgt (Tensor): Target sequence input to the decoder layer.
            memory (Tensor): Output of the encoder to be used in cross-attention.
            tgt_padding_mask (Tensor): Mask for the target sequence to ignore padding tokens.
            memory_padding_mask (Tensor): Mask for the encoder output to ignore padding tokens.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The output tensor, self-attention weights, and cross-attention weights.
        """
        sa, sa_weights = self.self_attention(
            source_query=tgt,
            source_key_value=tgt,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=tgt_padding_mask,
        )
        sa_norm = self.ln1(sa + tgt)
        ca, ca_weights = self.cross_attention(
            source_query=sa_norm,
            source_key_value=memory,
            source_query_padding_mask=tgt_padding_mask,
            source_key_value_padding_mask=memory_padding_mask,
        )
        ca_norm = self.ln2(ca + sa_norm)
        out = self.ln3(self.ffwd(ca_norm) + ca_norm)
        return out, sa_weights, ca_weights

class Transformer(nn.Module):
    """
    Implements the Transformer architecture for sequence-to-sequence tasks.

    This class combines the encoder and decoder along with initial embedding layers
    and final linear projection to implement the complete Transformer model.

    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimensionality of the embeddings and model.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network's hidden layer.
        n_encoder_layers (int): Number of layers in the encoder.
        n_decoder_layers (int): Number of layers in the decoder.
        dropout (float): Dropout rate.

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        encoder (nn.ModuleList): List of encoder layers.
        decoder (nn.ModuleList): List of decoder layers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PADDING_IDX)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, dropout, masked=False) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, dropout) for _ in range(n_decoder_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1) # gain=0.1 is the secret sauce to make it stable

    @torch.no_grad()
    def get_attention_weights(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieves attention weights from all encoder and decoder layers.

        Useful for visualization and analysis of the attention mechanism's focus during translation or other tasks.

        Args:
            src (Tensor): Source sequence tensor.
            tgt (Tensor): Target sequence tensor.
            src_padding_mask (Tensor): Mask for source sequence to ignore padding.
            tgt_padding_mask (Tensor): Mask for target sequence to ignore padding.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Attention weights from encoder self-attention, decoder self-attention, and decoder cross-attention layers.
        """
        enc_att_weights_all = []
        enc = self.positional_encoding(self.tok_emb(src.long()) * self.d_model**0.5)
        for layer in self.encoder:
            enc, enc_att_weights = layer(enc, src_padding_mask)
            enc_att_weights_all.append(enc_att_weights)
        enc_att_weights_all = torch.stack(enc_att_weights_all) # (n_layers, batch_size, n_heads, seq_len_q, seq_len_kv)
        enc_att_weights_all = enc_att_weights_all.permute(0, 2, 1, 3, 4) # (n_layers, n_heads, batch_size, seq_len_q, seq_len_kv)

        dec_self_att_weights_all = []
        enc_dec_weights_all = []
        dec = self.positional_encoding(self.tok_emb(tgt.long()) * self.d_model**0.5)
        for layer in self.decoder:
            dec, dec_self_att_weights, enc_dec_weights = layer(dec, enc, tgt_padding_mask, src_padding_mask)
            dec_self_att_weights_all.append(dec_self_att_weights)
            enc_dec_weights_all.append(enc_dec_weights)
        dec_self_att_weights_all = torch.stack(dec_self_att_weights_all)
        enc_dec_weights_all = torch.stack(enc_dec_weights_all)
        dec_self_att_weights_all = dec_self_att_weights_all.permute(0, 2, 1, 3, 4)
        enc_dec_weights_all = enc_dec_weights_all.permute(0, 2, 1, 3, 4)
        return enc_att_weights_all, dec_self_att_weights_all, enc_dec_weights_all

    def encode(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        """
        Encodes the source sequence.

        Applies embeddings, positional encoding, and processes the input through the encoder layers.

        Args:
            src (Tensor): Source sequence tensor.
            src_padding_mask (Tensor): Mask for source sequence to ignore padding.

        Returns:
            Tensor: Encoded representation of source sequence.
        """
        enc = self.positional_encoding(self.tok_emb(src.long()) * self.d_model**0.5)
        for layer in self.encoder:
            enc, _ = layer(enc, src_padding_mask)
        return enc

    def decode(
        self,
        tgt: Tensor,
        enc: Tensor,
        tgt_padding_mask: Tensor,
        src_padding_mask: Tensor
    ) -> Tensor:
        """
        Decodes the target sequence.

        Processes the target sequence (shifted right) through the decoder layers using the encoder's output.

        Args:
            tgt (Tensor): Target sequence tensor.
            enc (Tensor): Encoded source sequence from the encoder.
            tgt_padding_mask (Tensor): Mask for target sequence to ignore padding.
            src_padding_mask (Tensor): Mask for encoder output to ignore padding in cross-attention.

        Returns:
            Tensor: Decoded representation of target sequence.
        """
        dec = self.positional_encoding(self.tok_emb(tgt.long()) * self.d_model**0.5)
        for layer in self.decoder:
            dec, _, _ = layer(dec, enc, tgt_padding_mask, src_padding_mask)
        return dec

    def unembedding(self, dec: Tensor) -> Tensor:
        """
        Converts decoder output to logits over the vocabulary.

        This method projects the decoder output to the vocabulary space using the transpose of the embedding matrix.

        Args:
            dec (Tensor): Decoder output tensor.

        Returns:
            Tensor: Logits for each token in the target sequence.
        """
        logits = dec @ self.tok_emb.weight.T
        return logits

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor
    ) -> Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (Tensor): Input sequence to the encoder.
            tgt (Tensor): Target sequence for the decoder.
            src_padding_mask (Tensor): Mask for the encoder input to ignore padding tokens.
            tgt_padding_mask (Tensor): Mask for the decoder input to ignore padding tokens.

        Returns:
            Tensor: The output logits from the final linear layer.
        """
        enc = self.encode(src, src_padding_mask)
        dec = self.decode(tgt, enc, tgt_padding_mask, src_padding_mask)
        logits = self.unembedding(dec)
        return logits

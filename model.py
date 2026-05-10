"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # raise NotImplementedError
    d_k = Q.size(-1)  # Get the dimensionality of the keys
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # Compute scaled dot product
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))  # Apply mask
    attn_weights = F.softmax(attn_scores, dim=-1)  # Compute attention weights
    output = torch.matmul(attn_weights, V)  # Compute the attended output
    return output, attn_weights

# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 0)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # raise NotImplementedError
    return src.unsqueeze(1).unsqueeze(2) == pad_idx


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 0)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # Padding mask
    pad_mask = tgt.unsqueeze(1).unsqueeze(2) == pad_idx  # [batch, 1, 1, tgt_len]
    
    # Causal mask (look-ahead mask)
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
    
    # Combine: True where either PAD or future token
    combined_mask = pad_mask | causal_mask
    return combined_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)
        
        # Project Q, K, V and split into multiple heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, num_heads, seq_q, d_k]
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)    # [batch, num_heads, seq_k, d_k]
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, num_heads, seq_k, d_k]
        
        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)  # [batch, num_heads, seq_q, d_k]
        
        # Concatenate heads: [batch, num_heads, seq_q, d_k] → [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Positional Encoding as in "Attention Is All You Need", §3.5.
    
    Supports two modes:
    - 'sinusoidal' (default): Fixed sinusoidal encoding (non-trainable)
    - 'learned': Learned positional embeddings (trainable)

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
        mode     (str)  : 'sinusoidal' or 'learned' (default: 'sinusoidal').
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, mode: str = 'sinusoidal') -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode
        self.d_model = d_model
        self.max_len = max_len
        
        if mode == 'sinusoidal':
            self._init_sinusoidal(max_len, d_model)
        elif mode == 'learned':
            self._init_learned(max_len, d_model)
        else:
            raise ValueError(f"Unknown positional encoding mode: {mode}")
    
    def _init_sinusoidal(self, max_len: int, d_model: int) -> None:
        """Initialize sinusoidal positional encoding (non-trainable)."""
        # Create PE matrix: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                             (-math.log(10000.0) / d_model))  # [d_model/2]
        
        pe[:, 0::2] = torch.sin(position * div_term)    # even indices
        pe[:, 1::2] = torch.cos(position * div_term)    # odd indices
        
        # Register as buffer (not trainable parameter)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def _init_learned(self, max_len: int, d_model: int) -> None:
        """Initialize learned positional embeddings (trainable)."""
        # Learnable embedding for positions
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  positional_encoding[:, :seq_len, :]  

        """
        seq_len = x.size(1)
        
        if self.mode == 'sinusoidal':
            x = x + self.pe[:, :seq_len, :]
        elif self.mode == 'learned':
            # Get position indices [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)  # [1, seq_len]
            pos_embed = self.pos_embedding(positions)  # [1, seq_len, d_model]
            x = x + pos_embed
        
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Post-LayerNorm: x → attention → add & norm
        attn_output = self.self_attn(x, x, x, src_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Post-LayerNorm: x → ffn → add & norm
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Post-LayerNorm: Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        
        # Post-LayerNorm: Cross-attention with encoder memory
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        
        # Post-LayerNorm: Feed-forward
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        return x


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = None,
        checkpoint_drive_id: Optional[str] = None,
        pos_encoding_mode: str = 'sinusoidal',
    ) -> None:
        super().__init__()

        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_inv_vocab = None
        self.tgt_inv_vocab = None

        self._load_inference_assets()

        if src_vocab_size is None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(self.tgt_vocab)
        
        # Store config for checkpointing
        self.config = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model,
            'N': N,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'dropout': dropout,
            'pos_encoding_mode': pos_encoding_mode,
        }
        
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, mode=pos_encoding_mode)
        
        # Encoder and Decoder
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)
        
        # Output projection to vocabulary
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        self._load_or_initialize_weights(checkpoint_path, checkpoint_drive_id)

    def _load_inference_assets(self) -> None:
        """Load tokenizers and vocabularies required by infer() inside __init__."""
        from dataset import Multi30kDataset

        data = Multi30kDataset()
        data.build_vocab()

        self.src_tokenizer = data.tokenize_de
        self.tgt_tokenizer = data.tokenize_en
        self.src_vocab = data.de_vocab
        self.tgt_vocab = data.en_vocab
        self.src_inv_vocab = {i: t for t, i in data.de_vocab.items()}
        self.tgt_inv_vocab = {i: t for t, i in data.en_vocab.items()}

    def _load_or_initialize_weights(
        self,
        checkpoint_path: Optional[str],
        checkpoint_drive_id: Optional[str],
    ) -> None:
        """Load weights from local path or Google Drive; otherwise Xavier-init."""
        if checkpoint_path is None:
            checkpoint_path = os.getenv("A3_CHECKPOINT_PATH", "transformer_weights.pt")

        if checkpoint_drive_id is None:
            checkpoint_drive_id = os.getenv("A3_CHECKPOINT_DRIVE_ID")

        if (not os.path.exists(checkpoint_path)) and checkpoint_drive_id:
            gdown.download(id=checkpoint_drive_id, output=checkpoint_path, quiet=False)

        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.load_state_dict(state)
            return

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Embed and add positional encoding
        x = self.src_embed(src) * math.sqrt(self.config['d_model'])
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Embed and add positional encoding
        x = self.tgt_embed(tgt) * math.sqrt(self.config['d_model'])
        x = self.pos_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        if self.src_tokenizer is None or self.src_vocab is None:
            raise RuntimeError("Inference assets are not available.")
        
        device = next(self.parameters()).device
        
        # Tokenize source
        src_tokens = self.src_tokenizer(src_sentence)
        
        # Convert to indices
        unk_idx = self.src_vocab.get('<unk>', self.unk_idx)
        sos_idx = self.src_vocab.get('<sos>', self.sos_idx)
        eos_idx = self.tgt_vocab.get('<eos>', self.eos_idx)
        pad_idx = self.src_vocab.get('<pad>', self.pad_idx)
        
        src_eos_idx = self.src_vocab.get('<eos>', self.eos_idx)
        src_indices = [sos_idx] + [self.src_vocab.get(token, unk_idx) for token in src_tokens] + [src_eos_idx]
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
        
        # Create mask
        src_mask = make_src_mask(src_tensor, pad_idx).to(device)
        
        # Encode
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            
            # Decode greedily
            ys = torch.ones(1, 1, dtype=torch.long).fill_(sos_idx).to(device)
            max_len = 100
            
            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                
                # Get next token (greedy)
                next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_token], dim=1)
                
                # Stop if we generate <eos>
                if next_token.item() == eos_idx:
                    break
        
        # Convert indices to tokens
        pred_indices = ys.squeeze(0).cpu().tolist()
        pred_tokens = [
            self.tgt_inv_vocab.get(idx, '<unk>')
            for idx in pred_indices[1:]  # Skip <sos>
        ]
        
        # Remove <eos> and <pad> tokens
        if '<eos>' in pred_tokens:
            pred_tokens = pred_tokens[:pred_tokens.index('<eos>')]
        pred_tokens = [t for t in pred_tokens if t != '<pad>']
        
        # Detokenize and return
        return ' '.join(pred_tokens)


def test_infer(german_sentence: str = "ein kleines kind spielt im park") -> str:
    """Simple local smoke test for autograder-style single-sentence inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer().to(device)
    model.eval()
    return model.infer(german_sentence)

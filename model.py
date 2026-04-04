"""
ArmGPT Model - A simple GPT (Generative Pre-trained Transformer)

This is the brain of ArmGPT. It learns to predict the next character/token
in Armenian text. The architecture is the same as GPT-2, just smaller.

Key idea: The model reads a sequence of tokens and predicts what comes next.
           During training, it gets better at this prediction over time.

Architecture (each piece builds on the last):
    1. Token Embedding:   convert token IDs to vectors
    2. Position Embedding: tell the model where each token is in the sequence
    3. Transformer Blocks: the model "thinks" by letting tokens talk to each other
    4. Output Head:        predict the next token

A Transformer Block has two parts:
    - Self-Attention: each token looks at previous tokens to gather context
    - MLP (Feed-Forward): each token processes its gathered information
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Each token attends to (looks at) all previous tokens to gather context."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        # Key, Query, Value projections — all in one matrix for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # Causal mask: tokens can only look at previous positions, not future ones
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Query, Key, Value for all heads at once
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into multiple heads: (B, T, C) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention: how much should each token pay attention to each other token?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        # Gather information from the tokens we're attending to
        y = att @ v
        # Recombine heads: (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """A simple feed-forward network: each token processes its information."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)   # expand
        self.gelu = nn.GELU()                         # activation function
        self.c_proj = nn.Linear(4 * n_embd, n_embd)  # project back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One Transformer block: self-attention followed by MLP, with residual connections."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)  # normalize before attention
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)  # normalize before MLP
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        # Residual connection: add the output back to the input
        # This helps the model train by letting gradients flow easily
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    The full GPT language model.

    Takes a sequence of token IDs and predicts the next token at each position.
    """

    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            # Token embedding: token ID -> vector
            wte=nn.Embedding(vocab_size, n_embd),
            # Position embedding: position -> vector
            wpe=nn.Embedding(block_size, n_embd),
            # Dropout on embeddings
            drop=nn.Dropout(dropout),
            # Stack of transformer blocks
            blocks=nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            # Final layer normalization
            ln_f=nn.LayerNorm(n_embd),
        ))
        # Output head: vector -> vocabulary logits (one score per possible next token)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head
        # This is a common trick that improves quality and reduces parameters
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model initialized: {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights using small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass: predict next tokens.

        Args:
            idx: input token IDs, shape (batch_size, sequence_length)
            targets: target token IDs for computing loss (optional)

        Returns:
            logits: prediction scores for each position, shape (B, T, vocab_size)
            loss: cross-entropy loss (only if targets are provided)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Create position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Combine token and position embeddings
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # Final normalization
        x = self.transformer.ln_f(x)

        # Predict next token
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if we have targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens one at a time.

        Args:
            idx: starting token IDs, shape (1, T)
            max_new_tokens: how many new tokens to generate
            temperature: controls randomness (0.1 = safe, 1.0 = creative, 2.0 = wild)
            top_k: only sample from the top k most likely tokens (None = all)

        Returns:
            idx: the full sequence including generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens (the model's context window)
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Take only the last position's predictions
            logits = logits[:, -1, :] / temperature

            # Optionally only keep the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

"""
ArmGPT Model - A modern GPT with RMSNorm, SwiGLU, and RoPE.

Architecture:
    1. Token Embedding:   convert token IDs to vectors
    2. RoPE:              rotary position embeddings (no learned position table)
    3. Transformer Blocks: RMSNorm + Attention + SwiGLU MLP
    4. Output Head:        predict the next token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster than LayerNorm, no bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    """Precompute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to query/key tensors."""
    B, n_head, T, head_dim = x.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    # Split into pairs and rotate
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    """Self-attention with RoPE (no causal mask buffer needed — using F.scaled_dot_product_attention)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        # Precompute RoPE
        cos, sin = precompute_rope(self.head_dim, block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Use PyTorch's efficient attention (handles causal mask internally)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network — better than GELU, used by LLaMA/Mistral."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        # SwiGLU uses 8/3 * n_embd hidden dim (rounded to multiple of 64 for efficiency)
        hidden = int(8 / 3 * n_embd)
        hidden = ((hidden + 63) // 64) * 64  # round up to multiple of 64
        self.w1 = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, n_embd, bias=False)   # down
        self.w3 = nn.Linear(n_embd, hidden, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Transformer block: RMSNorm + Attention + SwiGLU MLP."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT language model with RMSNorm, RoPE, and SwiGLU."""

    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            drop=nn.Dropout(dropout),
            blocks=nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ]),
            ln_f=RMSNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model initialized: {n_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Token embeddings only — RoPE handles positions inside attention
        x = self.transformer.drop(self.transformer.wte(idx))

        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 stop_tokens=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_tokens and idx_next.item() in stop_tokens:
                break
        return idx

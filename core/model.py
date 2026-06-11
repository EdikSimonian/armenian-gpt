"""
ArmGPT Model - A modern GPT with RMSNorm, SwiGLU, and RoPE.

Architecture:
    1. Token Embedding:   convert token IDs to vectors
    2. RoPE:              rotary position embeddings (no learned position table)
    3. Transformer Blocks: RMSNorm + Attention + SwiGLU MLP
    4. Output Head:        predict the next token
"""

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


def apply_rope(x, cos, sin, pos_offset=0):
    """Apply rotary position embeddings to query/key tensors.

    pos_offset shifts the absolute position, for KV-cached incremental decoding
    where the new token sits at position = number of already-cached tokens.
    """
    B, n_head, T, head_dim = x.shape
    cos = (
        cos[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)
    )  # (1, 1, T, head_dim//2)
    sin = sin[pos_offset : pos_offset + T].unsqueeze(0).unsqueeze(0)
    # Split into pairs and rotate
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
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

    def forward(self, x, past_kv=None, pos_offset=0, use_cache=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE at the correct absolute positions (pos_offset..pos_offset+T)
        q = apply_rope(q, self.rope_cos, self.rope_sin, pos_offset)
        k = apply_rope(k, self.rope_cos, self.rope_sin, pos_offset)

        # KV cache: prepend previously-computed (already RoPE'd) keys/values.
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        new_kv = (k, v) if use_cache else None

        # Prefill (no cache) uses the built-in causal mask. Incremental decode
        # (past_kv set, single new query) attends to all cached keys, no mask.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_kv


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network — better than GELU, used by LLaMA/Mistral."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        # SwiGLU uses 8/3 * n_embd hidden dim (rounded to multiple of 64 for efficiency)
        hidden = int(8 / 3 * n_embd)
        hidden = ((hidden + 63) // 64) * 64  # round up to multiple of 64
        self.w1 = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, n_embd, bias=False)  # down
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

    def forward(self, x, past_kv=None, pos_offset=0, use_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv, pos_offset, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class GPT(nn.Module):
    """GPT language model with RMSNorm, RoPE, and SwiGLU."""

    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                drop=nn.Dropout(dropout),
                blocks=nn.ModuleList(
                    [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
                ),
                ln_f=RMSNorm(n_embd),
            )
        )
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

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.size()
        pos_offset = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert pos_offset + T <= self.block_size, (
            f"Sequence position {pos_offset + T} exceeds block_size {self.block_size}"
        )

        # Token embeddings only — RoPE handles positions inside attention
        x = self.transformer.drop(self.transformer.wte(idx))

        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past_kv=past, pos_offset=pos_offset, use_cache=use_cache)
            if use_cache:
                new_kvs.append(kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if use_cache:
            return logits, loss, new_kvs
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        stop_tokens=None,
        repetition_penalty=1.0,
    ):
        """Generate tokens autoregressively (KV-cached).

        A single prefill pass over the prompt builds a per-layer key/value cache;
        each subsequent step feeds only the newly sampled token and reuses the
        cache, so decoding is O(T) per token instead of re-running the full
        forward (O(T^2) overall). Output is identical to the un-cached loop.

        Args:
            repetition_penalty: 1.0 = no penalty (off). >1.0 discourages
                repeating tokens already in the context (CTRL-style penalty).
                Typical values: 1.1–1.3. Helps small LMs escape repetition loops.

        Note: batch size 1 is assumed when stop_tokens/repetition_penalty are
        used. Generation stops once the cache spans block_size positions (no
        sliding-window re-encode), which is well beyond typical chat lengths.
        """
        kvs = None
        for _ in range(max_new_tokens):
            if kvs is None:
                # Prefill: encode the whole (possibly truncated) prompt once.
                idx_cond = idx[:, -self.block_size :]
                logits, _, kvs = self(idx_cond, use_cache=True)
            else:
                # Incremental decode: feed only the newest token, reuse cache.
                logits, _, kvs = self(idx[:, -1:], past_kvs=kvs, use_cache=True)
            logits = logits[:, -1, :]

            # CTRL-style repetition penalty over the full running context.
            # Positive logits get divided (made smaller); negative logits get
            # multiplied (made more negative). Applied before temperature.
            if repetition_penalty != 1.0:
                seen = torch.unique(idx)
                seen_logits = logits[:, seen]
                seen_logits = torch.where(
                    seen_logits > 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty,
                )
                logits[:, seen] = seen_logits

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_tokens and idx_next.item() in stop_tokens:
                break
            # The cache spans absolute positions; once it fills block_size we
            # cannot extend further without a sliding-window re-encode.
            if kvs[0][0].size(2) >= self.block_size:
                break
        return idx

"""A small GPT in plain PyTorch with architectural optimizations:
- Weight Tying between token embeddings and language model head
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU in the MLP block
- Rotary Position Embeddings (RoPE) instead of absolute position embeddings
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 2048      # custom BPE tokenizer default
    block_size = 128
    n_layer = 4
    n_head = 6
    n_embd = 180
    dropout = 0.0
    tie_weights = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int(8 * cfg.n_embd / 3)
        self.w1 = nn.Linear(cfg.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # shape (end, dim // 2)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    # x shape: (B, n_head, T, head_dim)
    # cos, sin shape: (T, head_dim // 2)
    cos_full = torch.cat([cos, cos], dim=-1)  # (T, head_dim)
    sin_full = torch.cat([sin, sin], dim=-1)  # (T, head_dim)
    
    cos_full = cos_full[None, None, :, :].to(x.device)
    sin_full = sin_full[None, None, :, :].to(x.device)
    
    return x * cos_full + rotate_half(x) * sin_full


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        
        # Precompute RoPE freqs
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_freqs_cis(head_dim, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Weight tying
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)
        
        # Enforce strict parameter budget count limit < 1,980,000
        n_params = self.n_params()
        if n_params >= 1980000:
            raise ValueError(f"Model capacity check failed: {n_params:,} parameters exceeds limit of 1,980,000")

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        
        cos = self.cos[:T]
        sin = self.sin[:T]
        
        for blk in self.blocks:
            x = blk(x, cos, sin)
            
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # Tied weights are only counted once because they share the same Parameter object
        return sum(p.numel() for p in self.parameters())

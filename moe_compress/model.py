"""A small decoder-only Mixture-of-Experts language model in MLX.

Experts are evaluated densely and masked by the router weights. That is not how
a production kernel works, but it keeps the code short and means compute scales
with the number of experts, which is what the compression study measures.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class Config:
    vocab: int = 64
    dim: int = 128
    heads: int = 4
    layers: int = 4
    experts: int = 8
    top_k: int = 2
    hidden: int = 256


class RoutingStats:
    """Per-expert token counts and gate mass, filled during calibration."""

    def __init__(self, n_experts):
        self.count = np.zeros(n_experts)
        self.mass = np.zeros(n_experts)

    def update(self, inds, scores):
        inds, scores = np.array(inds), np.array(scores)
        for e in range(len(self.count)):
            hit = inds == e
            self.count[e] += hit.sum()
            self.mass[e] += (scores * hit).sum()


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, dim, hidden, n_experts, top_k):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.experts = [Expert(dim, hidden) for _ in range(n_experts)]
        self.top_k = top_k
        self.stats = None

    def route(self, x):
        logits = self.router(x)
        probs = mx.softmax(logits, axis=-1)
        picked = mx.argpartition(-mx.stop_gradient(logits), kth=self.top_k - 1, axis=-1)
        inds = picked[:, : self.top_k]
        scores = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)
        return inds, scores, probs

    def __call__(self, x, aux=None):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        inds, scores, probs = self.route(x)
        n_experts = len(self.experts)

        out = mx.zeros_like(x)
        for e, expert in enumerate(self.experts):
            weight = ((inds == e) * scores).sum(axis=-1)
            out = out + weight[:, None] * expert(x)

        if aux is not None:
            # Switch-style load-balancing term: E * sum_e f_e * P_e
            chosen = (inds[..., None] == mx.arange(n_experts)).astype(probs.dtype)
            frac = chosen.sum(axis=1).mean(axis=0) / self.top_k
            aux.append(n_experts * (frac * probs.mean(axis=0)).sum())
        if self.stats is not None:
            mx.eval(inds, scores)
            self.stats.update(inds, scores)
        return out.reshape(shape)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(dim // heads)

    def __call__(self, x):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5, mask="causal")
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.norm2 = nn.RMSNorm(cfg.dim)
        self.moe = MoE(cfg.dim, cfg.hidden, cfg.experts, cfg.top_k)

    def __call__(self, x, aux=None, trace=None):
        h = x + self.attn(self.norm1(x))
        out = h + self.moe(self.norm2(h), aux) if self.moe is not None else h
        if trace is not None:
            trace.append((x, h, out))
        return out


class MoELM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)

    def __call__(self, tokens, aux=None, trace=None):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, aux, trace)
        return self.head(self.norm(x))

    def moes(self):
        return [b.moe for b in self.blocks if b.moe is not None]

"""Expert-level compression: trimming, merging, slimming and quantization."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .model import MoE, RoutingStats


def collect_stats(model, batches):
    """Run calibration batches and return one RoutingStats per remaining MoE layer."""
    moes = model.moes()
    for moe in moes:
        moe.stats = RoutingStats(len(moe.experts))
    for tokens in batches:
        mx.eval(model(tokens))
    stats = [moe.stats for moe in moes]
    for moe in moes:
        moe.stats = None
    return stats


def _linear(weight):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    layer.weight = weight
    return layer


def _resolve(keep, total):
    n = int(round(keep * total)) if isinstance(keep, float) else int(keep)
    return max(1, min(total, n))


def _set_router(moe, rows):
    moe.router = _linear(moe.router.weight[mx.array(rows)])
    moe.top_k = min(moe.top_k, len(rows))


def trim_experts(model, stats, keep, strategy="usage"):
    """Remove the least useful experts in every layer.

    strategy: "usage" ranks by routed token count, "mass" by summed gate weight.
    """
    for moe, st in zip(model.moes(), stats):
        n = _resolve(keep, len(moe.experts))
        score = st.count if strategy == "usage" else st.mass
        rows = sorted(np.argsort(-score)[:n].tolist())
        moe.experts = [moe.experts[i] for i in rows]
        _set_router(moe, rows)
    return model


def merge_experts(model, stats, keep):
    """Fold dropped experts into their nearest kept expert.

    Anchors are the most-used experts. Each dropped expert joins the anchor whose
    router row it is most similar to, and the group's weights are averaged with
    usage as the weighting.
    """
    for moe, st in zip(model.moes(), stats):
        total = len(moe.experts)
        n = _resolve(keep, total)
        anchors = sorted(np.argsort(-st.count)[:n].tolist())
        router = np.array(moe.router.weight)
        unit = router / (np.linalg.norm(router, axis=1, keepdims=True) + 1e-8)

        groups = {a: [a] for a in anchors}
        for j in range(total):
            if j not in groups:
                best = max(anchors, key=lambda a: float(unit[a] @ unit[j]))
                groups[best].append(j)

        for a, members in groups.items():
            w = np.array([st.count[m] + 1e-3 for m in members])
            w = w / w.sum()
            for name in ("gate", "up", "down"):
                mixed = sum(
                    float(wi) * getattr(moe.experts[m], name).weight for wi, m in zip(w, members)
                )
                getattr(moe.experts[a], name).weight = mixed
        moe.experts = [moe.experts[a] for a in anchors]
        _set_router(moe, anchors)
    return model


def slim_experts(model, keep, multiple=32):
    """Shrink every expert's hidden width, keeping its highest-norm neurons."""
    for moe in model.moes():
        for expert in moe.experts:
            gate, up, down = expert.gate.weight, expert.up.weight, expert.down.weight
            importance = (
                mx.linalg.norm(gate, axis=1) * mx.linalg.norm(up, axis=1) * mx.linalg.norm(down, axis=0)
            )
            hidden = gate.shape[0]
            n = _resolve(keep, hidden)
            n = min(hidden, max(multiple, int(math.ceil(n / multiple)) * multiple))
            rows = mx.array(sorted(np.argsort(-np.array(importance))[:n].tolist()))
            expert.gate = _linear(gate[rows])
            expert.up = _linear(up[rows])
            expert.down = _linear(down[:, rows])
    return model


def quantize_experts(model, bits=4, group_size=32):
    """Quantize only the expert projections, leaving attention and routers dense."""
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, m: isinstance(m, nn.Linear) and "experts" in path,
    )
    return model

"""Depth-wise compression: skip whole MoE sublayers or whole transformer blocks.

Both rank candidates by how little they change their input, measured as one minus
the mean cosine similarity between a token's hidden state going in and coming out.
"""

import mlx.core as mx
import numpy as np


def _cosine_gap(a, b):
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    cos = (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-8)
    return float(1.0 - cos.mean())


def _influence(model, batches, pick):
    totals = np.zeros(len(model.blocks))
    for tokens in batches:
        trace = []
        mx.eval(model(tokens, trace=trace))
        for i, step in enumerate(trace):
            a, b = pick(step)
            totals[i] += _cosine_gap(a, b)
    return totals / len(batches)


def drop_moe_layers(model, batches, n):
    """Disable the MoE sublayer in the n blocks where it matters least."""
    live = [i for i, b in enumerate(model.blocks) if b.moe is not None]
    scores = _influence(model, batches, lambda s: (s[1], s[2]))
    for i in sorted(live, key=lambda i: scores[i])[:n]:
        model.blocks[i].moe = None
    return model


def drop_blocks(model, batches, n):
    """Remove the n whole blocks (attention and MoE) that matter least."""
    scores = _influence(model, batches, lambda s: (s[0], s[2]))
    remove = set(sorted(range(len(model.blocks)), key=lambda i: scores[i])[:n])
    model.blocks = [b for i, b in enumerate(model.blocks) if i not in remove]
    return model

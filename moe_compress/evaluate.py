import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def count_params(model):
    """Total stored parameters, counting packed quantized weights as their real size."""
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def nbytes(model):
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def perplexity(model, batches):
    total, count = 0.0, 0
    for tokens in batches:
        logits = model(tokens[:, :-1])
        loss = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="sum")
        mx.eval(loss)
        total += float(loss)
        count += tokens[:, 1:].size
    return math.exp(total / count)


def throughput(model, batch=16, length=128, iters=10, vocab=64):
    """Forward-pass tokens per second after one warm-up call."""
    tokens = mx.random.randint(0, vocab, (batch, length))
    mx.eval(model(tokens))
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(model(tokens))
    return batch * length * iters / (time.perf_counter() - start)

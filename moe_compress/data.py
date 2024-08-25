"""A synthetic language where experts have something to specialise in.

Token 0..modes-1 at the start of a sequence selects one of several sparse Markov
chains over the remaining vocabulary. The model has to read the mode token and
then apply that mode's transition table, so routing by mode is a natural solution.
"""

import math

import mlx.core as mx
import numpy as np


class MarkovMixture:
    def __init__(self, vocab=64, modes=8, branching=3, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab, self.modes = vocab, modes
        body = vocab - modes
        table = np.zeros((modes, vocab, vocab))
        for m in range(modes):
            for s in range(vocab):
                nxt = rng.choice(body, size=branching, replace=False) + modes
                table[m, s, nxt] = rng.dirichlet(np.ones(branching))
        self.cum = table.cumsum(-1)
        self.table = table
        self.rng = rng

    def entropy_floor(self):
        """Per-token cross-entropy (nats) of the true chain, ignoring the mode token."""
        p = self.table[self.table > 0]
        return float(-(p * np.log(p)).sum() / (self.modes * self.vocab))

    def sample(self, batch, length):
        mode = self.rng.integers(0, self.modes, batch)
        seq = np.zeros((batch, length), dtype=np.int32)
        seq[:, 0] = mode
        seq[:, 1] = self.rng.integers(self.modes, self.vocab, batch)
        for t in range(2, length):
            u = self.rng.random(batch)
            cum = self.cum[mode, seq[:, t - 1]]
            seq[:, t] = np.minimum((u[:, None] > cum).sum(-1), self.vocab - 1)
        return mx.array(seq)

    def batches(self, n, batch, length):
        return [self.sample(batch, length) for _ in range(n)]

    def floor_perplexity(self):
        return math.exp(self.entropy_floor())

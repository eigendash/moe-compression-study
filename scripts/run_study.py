"""Train a toy MoE LM, then compare compression recipes on it.

    python scripts/run_study.py            # train (cached) and print a table
    python scripts/run_study.py --retrain  # ignore the cached baseline
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_compress import (  # noqa: E402
    Config,
    MoELM,
    collect_stats,
    count_params,
    drop_blocks,
    drop_moe_layers,
    merge_experts,
    perplexity,
    quantize_experts,
    slim_experts,
    throughput,
    trim_experts,
)
from moe_compress.data import MarkovMixture  # noqa: E402
from moe_compress.evaluate import nbytes  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"
CFG = Config()
SEQ = 65


def train(steps, batch=64, lr=3e-3, aux_coef=0.01):
    task = MarkovMixture(CFG.vocab, seed=0)
    model = MoELM(CFG)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps), weight_decay=0.01)

    def loss_fn(model, tokens):
        aux = []
        logits = model(tokens[:, :-1], aux=aux)
        ce = nn.losses.cross_entropy(logits, tokens[:, 1:]).mean()
        return ce + aux_coef * sum(aux) / len(aux)

    step = nn.value_and_grad(model, loss_fn)
    for i in range(1, steps + 1):
        loss, grads = step(model, task.sample(batch, SEQ))
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if i % 100 == 0 or i == 1:
            print(f"step {i:4d}  loss {float(loss):.3f}")
    ART.mkdir(exist_ok=True)
    model.save_weights(str(ART / "baseline.safetensors"))


def fresh():
    model = MoELM(CFG)
    model.load_weights(str(ART / "baseline.safetensors"))
    return model


RECIPES = {
    "baseline": lambda m, st, cal: m,
    "trim 75% experts": lambda m, st, cal: trim_experts(m, st, 0.75),
    "trim 50% experts": lambda m, st, cal: trim_experts(m, st, 0.5),
    "merge to 50% experts": lambda m, st, cal: merge_experts(m, st, 0.5),
    "slim experts 50%": lambda m, st, cal: slim_experts(m, 0.5),
    "4-bit experts": lambda m, st, cal: quantize_experts(m, bits=4),
    "drop 1 MoE layer": lambda m, st, cal: drop_moe_layers(m, cal, 1),
    "drop 2 MoE layers": lambda m, st, cal: drop_moe_layers(m, cal, 2),
    "drop 1 block": lambda m, st, cal: drop_blocks(m, cal, 1),
    "trim 50% + slim 50% + 4-bit": lambda m, st, cal: quantize_experts(
        slim_experts(trim_experts(m, st, 0.5), 0.5), bits=4
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    if args.retrain or not (ART / "baseline.safetensors").exists():
        train(args.steps)

    task = MarkovMixture(CFG.vocab, seed=0)
    task.rng = np.random.default_rng(1234)  # held-out stream
    calibration = task.batches(8, 32, SEQ)
    held_out = task.batches(16, 32, SEQ)
    print(f"\nground-truth perplexity floor: {task.floor_perplexity():.2f}\n")

    rows, base_ppl, base_tps = [], None, None
    for name, recipe in RECIPES.items():
        model = fresh()
        stats = collect_stats(model, calibration)
        recipe(model, stats, calibration)
        ppl, tps = perplexity(model, held_out), throughput(model, vocab=CFG.vocab)
        base_ppl = base_ppl or ppl
        base_tps = base_tps or tps
        rows.append((name, count_params(model), nbytes(model), ppl, tps / base_tps))

    print("| recipe | params | MB | perplexity | rel. speed |")
    print("|---|---:|---:|---:|---:|")
    for name, params, size, ppl, speed in rows:
        print(f"| {name} | {params:,} | {size / 1e6:.2f} | {ppl:.2f} | {speed:.2f}x |")


if __name__ == "__main__":
    main()

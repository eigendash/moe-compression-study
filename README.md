# moe-compression-study

A small, readable study of Mixture-of-Experts compression that runs natively on
Apple silicon with [MLX](https://github.com/ml-explore/mlx). It trains a tiny
MoE language model on a synthetic task and then compares the main families of
compression techniques on it.

It is inspired by the paper *Towards Efficient Mixture of Experts: A Holistic
Study of Compression Techniques* (TMLR 2025,
[arXiv:2406.02500](https://arxiv.org/abs/2406.02500)), which studies expert
trimming, expert slimming, layer drop and block drop on Mixtral-8x7B.

**This is not a reproduction of the paper's results.** It is an independent
toy-scale implementation of the technique families, written from the paper's
description. The model has 3.4M parameters, not Mixtral's billions, so the numbers below say
something about this toy and nothing about Mixtral.

## What is implemented

| Family | Function | Idea |
|---|---|---|
| Expert trimming | `trim_experts` | Drop the least-routed experts and their router rows. |
| Expert merging | `merge_experts` | Fold each dropped expert into the most similar kept expert, weighted by usage. |
| Expert slimming | `slim_experts` | Shrink every expert's hidden width, keeping its highest-norm neurons. |
| Quantization | `quantize_experts` | Quantize only expert projections with `mlx.nn.quantize`. |
| Layer drop | `drop_moe_layers` | Skip the MoE sublayer in blocks where it changes the hidden state least. |
| Block drop | `drop_blocks` | Remove whole blocks ranked by the same similarity measure. |

Drop candidates are ranked by `1 - mean cosine(h_in, h_out)` over calibration
tokens. Trimming and merging use per-expert routing counts from the same data.

## The toy task

Sequences start with one of 8 mode tokens, which selects a sparse Markov chain
over the rest of the vocabulary. A model that routes by mode can specialise one
expert per mode. The true per-token perplexity of the chain is 2.28.

## Results

Trained for 800 steps on an M4 Pro (about 40 seconds). Perplexity is on held-out
sequences. Speed is forward-pass tokens per second relative to the baseline.

| recipe | params | MB | perplexity | rel. speed |
|---|---:|---:|---:|---:|
| baseline | 3,429,504 | 13.72 | 2.41 | 1.00x |
| trim 75% experts | 2,642,048 | 10.57 | 4.12 | 1.25x |
| trim 50% experts | 1,854,592 | 7.42 | 26.00 | 1.63x |
| merge to 50% experts | 1,854,592 | 7.42 | 18.94 | 1.62x |
| slim experts 50% | 1,856,640 | 7.43 | 2.44 | 1.49x |
| 4-bit experts | 873,600 | 3.49 | 2.41 | 0.96x |
| drop 1 MoE layer | 2,642,048 | 10.57 | 2.50 | 1.28x |
| drop 2 MoE layers | 1,854,592 | 7.42 | 3.26 | 1.72x |
| drop 1 block | 2,576,256 | 10.31 | 3.07 | 1.33x |
| trim 50% + slim 50% + 4-bit | 429,184 | 1.72 | 25.65 | 1.72x |

Things worth noticing:

- Slimming experts to half width costs almost nothing here, and 4-bit expert
  quantization is lossless at this scale. Trimming is the damaging one, because
  the task has exactly as many modes as experts, so removing an expert removes a
  mode. A real model with redundant experts would behave differently.
- Merging beats trimming at the same expert count, but only by a little.
- 4-bit experts shrink memory 4x without a speedup. At this size the
  dequantize cost cancels the savings. Speed gains from quantization need
  larger matrices.
- Parameter counts for the 4-bit row count packed integers, so compare the MB
  column for the real storage.
- Everything is one seed on one tiny task. Treat it as a demo.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_study.py            # trains once, caches to artifacts/
python scripts/run_study.py --retrain
```

Requires an Apple silicon Mac. The experts are evaluated densely and masked by
router weights, which keeps the code short but is not an efficient MoE kernel.

## Layout

```
moe_compress/
  model.py     MoE language model (router, SwiGLU experts, RoPE attention)
  compress.py  trimming, merging, slimming, quantization
  drop.py      layer and block drop
  evaluate.py  perplexity, throughput, parameter counts
  data.py      synthetic Markov-mixture task
scripts/run_study.py
tests/
```

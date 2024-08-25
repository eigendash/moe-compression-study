import mlx.core as mx
import numpy as np
import pytest

from moe_compress import (
    Config,
    MoELM,
    collect_stats,
    count_params,
    drop_blocks,
    drop_moe_layers,
    merge_experts,
    quantize_experts,
    slim_experts,
    trim_experts,
)

CFG = Config(vocab=32, dim=64, heads=2, layers=3, experts=4, top_k=2, hidden=64)


@pytest.fixture
def setup():
    mx.random.seed(0)
    model = MoELM(CFG)
    mx.eval(model.parameters())
    batches = [mx.random.randint(0, CFG.vocab, (4, 16)) for _ in range(3)]
    return model, batches


def logits(model, batches):
    return np.array(model(batches[0]))


def test_stats_cover_every_routed_token(setup):
    model, batches = setup
    stats = collect_stats(model, batches)
    tokens = sum(b.size for b in batches)
    assert len(stats) == CFG.layers
    for st in stats:
        assert st.count.sum() == tokens * CFG.top_k
        assert st.mass.sum() == pytest.approx(tokens, rel=1e-4)
    assert all(m.stats is None for m in model.moes())


def test_trim_keeping_everything_is_identity(setup):
    model, batches = setup
    before = logits(model, batches)
    trim_experts(model, collect_stats(model, batches), CFG.experts)
    np.testing.assert_allclose(logits(model, batches), before, atol=1e-5)


def test_trim_removes_experts_and_router_rows(setup):
    model, batches = setup
    trim_experts(model, collect_stats(model, batches), 0.5)
    for moe in model.moes():
        assert len(moe.experts) == 2
        assert moe.router.weight.shape[0] == 2
    assert logits(model, batches).shape == (4, 16, CFG.vocab)


def test_trim_clamps_top_k(setup):
    model, batches = setup
    trim_experts(model, collect_stats(model, batches), 1)
    assert all(m.top_k == 1 for m in model.moes())
    assert np.isfinite(logits(model, batches)).all()


def test_merge_reduces_experts_and_keeps_output_finite(setup):
    model, batches = setup
    merge_experts(model, collect_stats(model, batches), 2)
    assert all(len(m.experts) == 2 for m in model.moes())
    assert np.isfinite(logits(model, batches)).all()


def test_slim_full_keep_is_identity_and_half_shrinks(setup):
    model, batches = setup
    before = logits(model, batches)
    slim_experts(model, 1.0)
    np.testing.assert_allclose(logits(model, batches), before, atol=1e-5)
    params = count_params(model)
    slim_experts(model, 0.5)
    assert count_params(model) < params
    assert model.blocks[0].moe.experts[0].gate.weight.shape[0] == 32


def test_quantize_stays_close_to_dense(setup):
    model, batches = setup
    before = logits(model, batches)
    quantize_experts(model, bits=8, group_size=32)
    after = logits(model, batches)
    assert np.abs(after - before).max() < 0.2 * np.abs(before).max()


def test_drop_moe_layers_and_blocks(setup):
    model, batches = setup
    drop_moe_layers(model, batches, 2)
    assert len(model.moes()) == 1 and len(model.blocks) == 3
    drop_blocks(model, batches, 1)
    assert len(model.blocks) == 2
    assert logits(model, batches).shape == (4, 16, CFG.vocab)

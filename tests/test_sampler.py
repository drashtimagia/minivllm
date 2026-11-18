import torch

from minivllm.sampler import sample
from minivllm.sequence import SamplingParams


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def test_greedy_picks_argmax() -> None:
    logits = torch.tensor([[0.1, 3.0, 0.2], [5.0, 1.0, 0.0]])
    out = sample(logits, [SamplingParams(temperature=0.0)] * 2, _generator())
    assert out.tolist() == [1, 0]


def test_top_p_keeps_smallest_covering_set() -> None:
    # Probabilities 0.5, 0.3, 0.2: top_p=0.6 needs the first two tokens, never the third.
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]])).repeat(2000, 1)
    params = [SamplingParams(temperature=1.0, top_p=0.6)] * 2000
    out = sample(logits, params, _generator())
    assert set(out.tolist()) == {0, 1}


def test_tiny_top_p_is_greedy() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.5]]).repeat(200, 1)
    out = sample(logits, [SamplingParams(temperature=1.0, top_p=1e-6)] * 200, _generator())
    assert set(out.tolist()) == {1}


def test_temperature_sharpens_distribution() -> None:
    logits = torch.tensor([[1.0, 0.0]]).repeat(4000, 1)
    hot = sample(logits, [SamplingParams(temperature=2.0)] * 4000, _generator())
    cold = sample(logits, [SamplingParams(temperature=0.25)] * 4000, _generator())
    assert (cold == 0).float().mean() > (hot == 0).float().mean() + 0.2


def test_mixed_batch_respects_each_row() -> None:
    logits = torch.tensor([[0.0, 9.0], [0.0, 0.0]]).repeat(500, 1)
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0)] * 500
    out = sample(logits, params, _generator())
    assert (out[0::2] == 1).all()
    assert set(out[1::2].tolist()) == {0, 1}

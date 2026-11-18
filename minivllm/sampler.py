import torch

from minivllm.sequence import SamplingParams


def sample(
    logits: torch.Tensor, params: list[SamplingParams], generator: torch.Generator
) -> torch.Tensor:
    """Pick one token per row of logits [batch, vocab], each row with its own params."""
    logits = logits.float()
    greedy_tokens = logits.argmax(dim=-1)
    if all(p.greedy for p in params):
        return greedy_tokens
    # Greedy rows get temperature 1 here only to avoid dividing by zero; their sampled
    # token is discarded below in favour of the argmax.
    temperatures = torch.tensor(
        [p.temperature if not p.greedy else 1.0 for p in params], device=logits.device
    )
    top_ps = torch.tensor([p.top_p for p in params], device=logits.device)
    probs = torch.softmax(logits / temperatures[:, None], dim=-1)
    sampled = _sample_top_p(probs, top_ps, generator)
    is_greedy = torch.tensor([p.greedy for p in params], device=logits.device)
    return torch.where(is_greedy, greedy_tokens, sampled)


def _sample_top_p(
    probs: torch.Tensor, top_ps: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    # Keep the smallest prefix of tokens whose mass reaches top_p: a token is dropped when
    # the mass strictly before it already covers top_p. The top token is always kept.
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs.masked_fill(mass_before >= top_ps[:, None], 0.0)
    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_ids.gather(-1, choice).squeeze(-1)

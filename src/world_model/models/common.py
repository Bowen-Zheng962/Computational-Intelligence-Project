from __future__ import annotations

import torch


def normal_parameters(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_std = raw.chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)


def reparameterize(
    mean: torch.Tensor, log_std: torch.Tensor, sample: bool
) -> torch.Tensor:
    if not sample:
        return mean
    return mean + torch.randn_like(mean) * log_std.exp()


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    posterior_var = (2.0 * posterior_log_std).exp()
    prior_var = (2.0 * prior_log_std).exp()
    kl = (
        prior_log_std
        - posterior_log_std
        + (posterior_var + (posterior_mean - prior_mean).pow(2)) / (2.0 * prior_var)
        - 0.5
    )
    return kl.sum(dim=-1)


def empty_regularizer(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros(reference.shape[:2], device=reference.device, dtype=reference.dtype)

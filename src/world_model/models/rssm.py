from __future__ import annotations

import torch
from torch import nn

from .common import diagonal_gaussian_kl, empty_regularizer, normal_parameters, reparameterize


class RecurrentStateSpaceModel(nn.Module):
    """Action-conditioned RSSM with posterior training and prior-only evaluation."""

    def __init__(
        self,
        history_dim: int,
        future_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        step_seconds: float = 1.0,
    ):
        super().__init__()
        del step_seconds
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.history_encoder = nn.GRU(history_dim, hidden_dim, batch_first=True)
        self.future_encoder = nn.Sequential(nn.Linear(future_dim, hidden_dim), nn.ReLU())
        self.initial_latent = nn.Linear(hidden_dim, 2 * latent_dim)
        self.cell = nn.GRUCell(hidden_dim + latent_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.posterior = nn.Linear(hidden_dim + 1, 2 * latent_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _rollout(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        target: torch.Tensor | None,
        sample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, history_hidden = self.history_encoder(history)
        hidden = history_hidden[-1]
        initial_mean, initial_log_std = normal_parameters(self.initial_latent(hidden))
        previous_latent = reparameterize(initial_mean, initial_log_std, sample)
        predictions = []
        kl_terms = []
        latent_states = []

        for step in range(future.shape[1]):
            action = self.future_encoder(future[:, step])
            hidden = self.cell(torch.cat([action, previous_latent], dim=-1), hidden)
            prior_mean, prior_log_std = normal_parameters(self.prior(hidden))
            if target is not None:
                post_mean, post_log_std = normal_parameters(
                    self.posterior(
                        torch.cat([hidden, target[:, step : step + 1]], dim=-1)
                    )
                )
                latent = reparameterize(post_mean, post_log_std, sample)
                kl = diagonal_gaussian_kl(
                    post_mean, post_log_std, prior_mean, prior_log_std
                )
            else:
                latent = reparameterize(prior_mean, prior_log_std, sample)
                kl = torch.zeros_like(prior_mean[:, 0])
            predictions.append(
                self.head(torch.cat([hidden, action, latent], dim=-1)).squeeze(-1)
            )
            kl_terms.append(kl)
            latent_states.append(latent)
            previous_latent = latent

        return (
            torch.stack(predictions, dim=1),
            torch.stack(kl_terms, dim=1),
            torch.stack(latent_states, dim=1),
        )

    def forward(
        self,
        history: torch.Tensor,
        future: torch.Tensor,
        physics: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        del physics, dt
        prediction, kl, latent = self._rollout(history, future, target, sample)
        zero = empty_regularizer(future)
        return {
            "prediction": prediction,
            "kl": kl,
            "physics": zero,
            "dynamics": zero,
            "latent": latent,
        }


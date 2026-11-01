"""Factory exposing only the original recurrent state-space predictor."""
from .rssm import RecurrentStateSpaceModel


def build_model(name, history_dim, future_dim, hidden_dim=64, latent_dim=16, step_seconds=5):
    if name != "rssm":
        raise ValueError("This delivery contains only the original RSSM backbone.")
    return RecurrentStateSpaceModel(history_dim, future_dim, hidden_dim, latent_dim, step_seconds)

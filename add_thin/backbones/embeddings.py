import numpy as np
import torch
import torch.nn as nn


class NyquistFrequencyEmbedding(nn.Module):
    """
    Sine-cosine embedding for timesteps that scales from 1/8 to a (< 1) multiple of
    the Nyquist frequency.

    We choose 1/8 as the slowest frequency so that the slowest-varying embedding varies
    roughly lineary across [0, 2pi] as the relative error between x and sin(x) on [0,
    2pi / 8] is at most 2.5%. The Nyquist frequency is the largest frequency that one
    can sample at T steps without aliasing, so one could assume that to be a great
    choice for the highest frequency but sampling sine and cosine at the Nyquist
    frequency would result in constant (and therefore uninformative) 1 and 0 features,
    so we Nyquist/2 is a better choice. However, Nyquist/2 (which is T/2) leads to the
    evaluation points of the fastest varying points to overlap, so that those features
    would only take a small number of values, such as 2 or 4. In combination with the
    other points, these embeddings would of course still be distinguishable but by
    choosing an irrational fastest frequency, we can get unique embeddings also in the
    fastest-varying dimension for all timepoints. We choose arbitrarily 1/phi where phi
    is the golden ratio.

    Parameters
    ----------
    dim : int
        Number of dimensions of the embedding
    timesteps : int
        Number of timesteps to embed
    """

    def __init__(self, dim: int, timesteps: int | float) -> None:
        super().__init__()

        assert dim % 2 == 0

        T = timesteps
        k = dim // 2

        # Nyquist frequency for T samples per cycle
        nyquist_frequency = T / 2

        golden_ratio = (1 + np.sqrt(5)) / 2
        frequencies = np.geomspace(
            1 / 8, nyquist_frequency / (2 * golden_ratio), num=k
        )

        # Sample every frequency twice, once shifted by pi/2 to get cosine
        scale = np.repeat(2 * np.pi * frequencies / timesteps, 2)
        bias = np.tile(np.array([0, np.pi / 2]), k)

        self.register_buffer(
            "scale",
            torch.from_numpy(scale.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "bias", torch.from_numpy(bias.astype(np.float32)), persistent=False
        )

    def forward(self, t) -> torch.Tensor:
        return torch.addcmul(self.bias, self.scale, t[..., None]).sin()

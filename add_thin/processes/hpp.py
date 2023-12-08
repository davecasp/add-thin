from typing import Union

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from add_thin.data import Batch

patch_typeguard()  # use before @typechecked


@typechecked
def generate_hpp(
    tmax: TensorType,
    n_sequences: int,
    intensity: Union[TensorType, None] = None,
) -> Batch:
    """
    Generate a batch of sequences from a homogeneous Poisson process on [0,T].

    Parameters
    ----------
    tmax : TensorType
        Maximum time of the sequence
    n_sequences : int
        Number of sequences to generate
    intensity : Union[TensorType, None], optional
        Intensity of the process, set to 1 if None, by default None

    Returns
    -------
    Batch
        Batch of generated sequences
    """
    device = tmax.device
    if intensity is None:
        intensity = torch.ones(n_sequences, device=device)

    # Get number of samples
    n_samples = torch.poisson(tmax * intensity)
    max_samples = int(torch.max(n_samples).item()) + 1

    # Sample times
    times = torch.rand((n_sequences, max_samples), device=device) * tmax

    # Mask for padding events
    mask = (
        torch.arange(0, max_samples, device=device)[None, :]
        < n_samples[:, None]
    )
    times = times * mask

    assert (mask.sum(-1) == n_samples).all(), "wrong number of samples"
    return Batch.remove_unnescessary_padding(
        time=times, mask=mask, tmax=tmax, kept=None
    )

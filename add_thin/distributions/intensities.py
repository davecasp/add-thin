from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import warnings
from torch.distributions import MixtureSameFamily
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from add_thin.data import Batch
from add_thin.distributions.densities import DISTRIBUTIONS

patch_typeguard()


@typechecked
class MixtureIntensity(nn.Module):
    """
    Class parameterizing the intensity function as a weighted mixture of distributions.

    Parameters:
    ----------
    n_components : int, optional
        Number of components to use in the mixture, by default 10
    embedding_size : int, optional
        Size of the event embedding, by default 128
    distribution : str, optional
        Distribution to use for the components, by default "normal"

    """

    def __init__(
        self,
        n_components: int = 10,
        embedding_size: int = 128,
        distribution: str = "normal",
    ) -> None:
        super().__init__()

        assert (
            distribution in DISTRIBUTIONS.keys()
        ), f"{distribution} not in {DISTRIBUTIONS.keys()}"
        self.w_activation = torch.nn.Softplus()
        self.distribution = DISTRIBUTIONS[distribution]

        # Parallel compute parameters weight, mu and sigma for n components with one MLP
        self.n_components = n_components
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 3 * n_components),
        )
        self.rejections_sample_multiple = 2

    def get_intensity_parameters(
        self,
        x_n: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """
        Compute the parameters of the intensity function.

        Parameters:
        ----------
        x_n : Batch
            Batch of event sequences to condition on
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time

        Returns:
        -------
        location, scale, weight: List[TensorType]
            The parameters of the intensity function
        """

        # Compute masked mean over sequence (zero padded)
        n_events = x_n.mask.sum(-1)
        seq_emb = event_emb.sum(1) / torch.clamp(n_events[..., None], min=1)

        parameters = self.mlp(torch.cat([seq_emb, dif_time_emb], dim=-1))
        return torch.split(
            parameters,
            [self.n_components, self.n_components, self.n_components],
            dim=-1,
        )

    def get_distribution(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
        L,
    ) -> Tuple[D.MixtureSameFamily, TensorType[float, "batch"]]:
        """
        Instantiate the mixture-distribution parameterizing the intensity function.

        Parameters:
        ----------
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        x_n : Batch
            Batch of event sequences to condition on
        L : int
            Maximum sequence length

        Returns:
        -------
        density, cumulative_intensity: Tuple[D.MixtureSameFamily, TensorType[float, "batch"]]
            The distribution and the cumulative intensity
        """
        location, scale, weight = self.get_intensity_parameters(
            x_n=x_n,
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
        )

        # Include the number of events in x_n for the cumulative intensity
        weight = self.w_activation(weight)
        cumulative_intensity = (weight).sum(-1) * (x_n.mask.sum(-1) + 1)

        # Probs is normalized to sum to 1
        mixture_dist = D.Categorical(probs=weight.unsqueeze(1).repeat(1, L, 1))

        # Distribution parameters are the same for each sequence element
        component_dist = self.distribution(
            location.unsqueeze(1).repeat(1, L, 1),
            scale.unsqueeze(1).repeat(1, L, 1),
        )
        return (
            MixtureSameFamily(mixture_dist, component_dist),
            cumulative_intensity,
        )

    def log_likelihood(
        self,
        x_0: Batch,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        x_n: Batch,
    ) -> TensorType[float, "batch"]:
        """
        Compute the log-likelihood of the event sequences.

        Parameters:
        ----------
        x_0 : Batch
            Batch of event sequences
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        log_likelihood: TensorType[float, "batch"]
            The log-likelihood of the event sequences
        """
        density, cif = self.get_distribution(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n,
            L=x_0.seq_len,
        )

        # Normalize event time to [0, 1]
        x = x_0.time / x_0.tmax

        # Compute log-intensity with re-weighting
        log_intensity = (
            (density.log_prob(x) + torch.log(cif)[..., None]) * x_0.mask
        ).sum(-1)

        # Compute CIF for normalization
        cdf = density.cdf(torch.ones_like(x)).mean(1)
        cif = cif * cdf  # Rescale between 0 and T

        return log_intensity - cif

    def sample(
        self,
        event_emb: TensorType[float, "batch", "seq", "embedding"],
        dif_time_emb: TensorType[float, "batch", "embedding"],
        n_samples: int,
        x_n: Batch,
    ) -> Batch:
        """
        Sample event sequences from the intensity function.

        Parameters:
        ----------
        event_emb : TensorType[float, "batch", "seq", "embedding"]
            Context embedding of the events
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        n_samples : int
            Number of samples to draw
        x_n : Batch
            Batch of event sequences to condition on

        Returns:
        -------
        Batch
            The sampled event sequences
        """
        tmax = x_n.tmax
        density, cif = self.get_distribution(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_n=x_n,
            L=1,
        )

        # Get number of points per sample sequence from CIF
        count_distribution = D.Poisson(
            rate=cif
            * density.cdf(
                torch.ones(n_samples, 1, device=event_emb.device)
            ).squeeze()
        )
        sequence_len = (
            count_distribution.sample((n_samples,)).squeeze()
        ).long()

        # TODO implement smarter truncated normal, without rejection sampling.
        max_seq_len = sequence_len.max()

        while True:
            times = (
                density.sample(
                    ((max_seq_len + 1) * self.rejections_sample_multiple,)
                )
                .squeeze(-1)
                .T
                * tmax
            )

            # Reject if not in [0, tmax]
            inside = torch.logical_and(times <= tmax, times >= 0)
            sort_idx = torch.argsort(
                inside.int(), stable=True, descending=True, dim=-1
            )
            inside = torch.take_along_dim(inside, sort_idx, dim=-1)[
                :, :max_seq_len
            ]
            times = torch.take_along_dim(times, sort_idx, dim=-1)[
                :, :max_seq_len
            ]

            # Randomly mask out events exceeding the actual sequence length
            mask = (
                torch.arange(0, times.shape[-1], device=times.device)[None, :]
                < sequence_len[:, None]
            )
            mask = mask * inside

            if (mask.sum(-1) == sequence_len).all():
                break
            else:
                self.rejections_sample_multiple += 1
                warnings.warn(
                    f"""
Rejection sampling multiple increased to {self.rejections_sample_multiple}, as not enough event times were inside [0, tmax].
""".strip()
                )

        times = times * mask

        return Batch.remove_unnescessary_padding(
            time=times, mask=mask, tmax=tmax, kept=None
        )

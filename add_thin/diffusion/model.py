import math
import torch
import torch.nn as nn

from typing import Tuple
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from add_thin.data import Batch
from add_thin.backbones.cnn import CNNSeqEmb
from add_thin.backbones.embeddings import NyquistFrequencyEmbedding
from add_thin.processes.hpp import generate_hpp
from add_thin.diffusion.utils import betas_for_alpha_bar

patch_typeguard()


@typechecked
class DiffusionModell(nn.Module):
    """
    Base class for diffusion models.

    Parameters
    ----------
    steps : int, optional
        Number of diffusion steps, by default 100
    """

    def __init__(self, steps: int = 1000) -> None:
        super().__init__()
        self.steps = steps

        # Cosine beta schedule
        beta = betas_for_alpha_bar(
            steps,
            lambda n: math.cos((n + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

        # Compute alpha and alpha_cumprod
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

        # Compute thinning probabilities for posterior and shift it to be indexed with n-1
        add_remove = (1 - self.alpha_cumprod)[:-1] * beta[1:]
        alpha_x0_kept = (self.alpha_cumprod[:-1] - self.alpha_cumprod[1:]) / (
            1 - self.alpha_cumprod[1:]
        )
        alpha_xn_kept = (
            (self.alpha - self.alpha_cumprod) / (1 - self.alpha_cumprod)
        )[1:]

        self.register_buffer("alpha_x0_kept", alpha_x0_kept)
        self.register_buffer("alpha_xn_kept", alpha_xn_kept)
        self.register_buffer("add_remove", add_remove)


@typechecked
class AddThin(DiffusionModell):
    """
    Implementation of AddThin (Add and Thin: Diffusion for Temporal Point Processes).

    Parameters
    ----------
    classifier_model : nn.Module
        Model for predicting the intersection of x_0 and x_n from x_n
    intensity_model : nn.Module
        Model for predicting the intensity of x_0 without x_n
    max_time : float
        T of the temporal point process
    n_max : int, optional
        Maximum number of events, by default 100
    steps : int, optional
        Number of diffusion steps, by default 100
    hidden_dims : int, optional
        Hidden dimensions of the models, by default 128
    emb_dim : int, optional
        Embedding dimensions of the models, by default 32
    encoder_layer : int, optional
        Number of encoder layers, by default 4
    kernel_size : int, optional
        Kernel size of the CNN, by default 16
    forecast : None, optional
        If not None, will turn the model into a conditional one for forecasting
    """

    def __init__(
        self,
        classifier_model,
        intensity_model,
        max_time: float,
        n_max: int = 100,
        steps: int = 100,
        hidden_dims: int = 128,
        emb_dim: int = 32,
        encoder_layer: int = 4,
        kernel_size: int = 16,
        forecast=None,
    ) -> None:
        super().__init__(steps)
        # Set models parametrizing the approximate posterior
        self.classifier_model = classifier_model
        self.intensity_model = intensity_model

        self.n_max = n_max

        # Init forecast settings
        if forecast:
            self.forecast = True
            self.history_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim,
                batch_first=True,
            )
            self.history_mlp = nn.Sequential(
                nn.Linear(2 * emb_dim, hidden_dims), nn.ReLU()
            )
            self.forecast_window = forecast
        else:
            self.forecast = False
            self.history = None

        self.set_encoders(
            hidden_dims=hidden_dims,
            max_time=max_time,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            kernel_size=kernel_size,
            steps=steps,
        )

    def set_encoders(
        self,
        hidden_dims: int,
        max_time: float,
        emb_dim: int,
        encoder_layer: int,
        kernel_size: int,
        steps: int,
    ) -> None:
        """
        Set the encoders for the model.

        Parameters
        ----------
        hidden_dims : int
            Hidden dimensions of the models
        max_time : float
            T of the temporal point process
        emb_dim : int
            Embedding dimensions of the models
        encoder_layer : int
            Number of encoder layers
        kernel_size : int
            Kernel size of the CNN
        steps : int
            Number of diffusion steps
        """
        # Event time encoder
        position_emb = NyquistFrequencyEmbedding(
            dim=emb_dim // 2, timesteps=max_time
        )
        self.time_encoder = nn.Sequential(position_emb)

        # Diffusion time encoder
        position_emb = NyquistFrequencyEmbedding(dim=emb_dim, timesteps=steps)
        self.diffusion_time_encoder = nn.Sequential(
            position_emb,
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Event sequence encoder
        self.sequence_encoder = CNNSeqEmb(
            emb_layer=encoder_layer,
            input_dim=hidden_dims,
            emb_dims=hidden_dims,
            kernel_size=kernel_size,
        )

    def set_history(self, batch: Batch) -> None:
        """
        Set the history to condition the model.

        Parameters
        ----------
        batch : Batch
            Batch of data
        """
        B, L = batch.time.shape

        # Encode event times
        time_emb = self.time_encoder(
            torch.cat(
                [batch.time.unsqueeze(-1), batch.tau.unsqueeze(-1)], dim=-1
            )
        ).reshape(B, L, -1)

        # Compute history embedding
        embedding = self.history_encoder(time_emb)[0]

        # Index relative to time and set history
        index = (batch.mask.sum(-1).long() - 1).unsqueeze(-1).unsqueeze(-1)
        gather_index = index.repeat(1, 1, embedding.shape[-1])
        self.history = embedding.gather(1, gather_index).squeeze(-2)

    def compute_emb(
        self, n: TensorType[torch.long, "batch"], x_n: Batch
    ) -> Tuple[
        TensorType["batch", "embedding"],
        TensorType["batch", "sequence", "embedding"],
        TensorType["batch", "sequence", "embedding"],
    ]:
        """
        Get the embeddings of x_n.

        Parameters
        ----------
        n : TensorType[torch.long, "batch"]
            Diffusion time step
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType["batch", "embedding"],
            TensorType["batch", "sequence", "embedding"],
            TensorType["batch", "sequence", "embedding"],
        ]
            Diffusion time embedding, event time embedding, event sequence embedding
        """
        B, L = x_n.batch_size, x_n.seq_len

        # embed diffusion and process time
        dif_time_emb = self.diffusion_time_encoder(n)

        # Condition ADD-THIN on history by adding it to the diffusion time embedding
        if self.forecast:
            dif_time_emb = self.history_mlp(
                torch.cat([self.history, dif_time_emb], dim=-1)
            )

        # Embed event and interevent time
        time_emb = self.time_encoder(
            torch.cat([x_n.time.unsqueeze(-1), x_n.tau.unsqueeze(-1)], dim=-1)
        ).reshape(B, L, -1)

        # Embed event sequence and mask out
        event_emb = self.sequence_encoder(time_emb)
        event_emb = event_emb * x_n.mask[..., None]

        return (
            dif_time_emb,
            time_emb,
            event_emb,
        )

    def get_n(self, shape, device, min=None, max=None) -> TensorType[int]:
        """
        Uniformly sample n, i.e., the diffusion time step per sequence.

        Parameters
        ----------
        shape :
            Shape of the tensor
        device :
            Device of the tensor
        min : None, optional
            Minimum value of n, by default None
        max : None, optional
            Maximum value of n, by default None

        Returns
        -------
        TensorType[int]
            Sampled n
        """
        if min is None or max is None:
            min = 0
            max = self.steps
        return torch.randint(
            min,
            max,
            size=shape,
            device=device,
            dtype=torch.long,
        )

    def noise(
        self, x_0: Batch, n: TensorType[torch.long, "batch"]
    ) -> Tuple[Batch, Batch]:
        """
        Sample x_n from x_0 by applying the noising process.

        Parameters
        ----------
        x_0 : Batch
            Batch of data
        n : TensorType[torch.long, "batch"]
            Number of noise steps

        Returns
        -------
        Tuple[Batch, Batch]
            x_n and thinned x_0
        """
        # Thin x_0
        x_0_kept, x_0_thinned = x_0.thin(alpha=self.alpha_cumprod[n])

        # Superposition with HPP (add)
        hpp = generate_hpp(
            tmax=x_0.tmax,
            n_sequences=len(x_0),
            intensity=1 - self.alpha_cumprod[n],
        )
        x_n = x_0_kept.add_events(hpp)

        return x_n, x_0_thinned

    def forward(
        self, x_0: Batch
    ) -> Tuple[
        TensorType[float, "batch", "sequence_x_n"],
        TensorType[float, "batch"],
        Batch,
    ]:
        """
        Forward pass to train the model, i.e., predict x_0 from x_n.

        Parameters
        ----------
        x_0 : Batch
            Batch of data

        Returns
        -------
        Tuple[
            TensorType[float, "batch", "sequence_x_n"],
            TensorType[float, "batch"],
            Batch,
        ]
            classification logits, log likelihood of x_0 without x_n, noised data
        """
        # Uniformly sample n
        n = self.get_n(
            min=0,
            max=self.steps,
            shape=(len(x_0),),
            device=x_0.time.device,
        )
        # Noise x_0 to get x_n
        x_n, x_0_thin = self.noise(x_0=x_0, n=n)

        # Embed x_n
        (dif_time_emb, time_emb, event_emb) = self.compute_emb(n=n, x_n=x_n)

        # Predict x_0 from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb,
            time_emb=time_emb,
            event_emb=event_emb,
        )

        # Evaluate intensity of thinned x_0
        log_like_x_0 = self.intensity_model.log_likelihood(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            x_0=x_0_thin,
            x_n=x_n,
        )

        return x_n_and_x_0_logits, log_like_x_0, x_n

    def sample(self, n_samples: int, tmax) -> Batch:
        """
        Sample x_0 from ADD-THIN starting from x_N.

        Parameters
        ----------
        n_samples : int
            Number of samples
        tmax : float
            T of the temporal point process
        begin_forecast : None, optional
            Beginning of the forecast, by default None
        end_forecast : None, optional
            End of the forecast, by default None

        Returns
        -------
        Batch
            Sampled x_0s
        """
        # Init x_N by sampling from HPP
        x_N = generate_hpp(tmax=tmax, n_sequences=n_samples)
        x_n_1 = x_N

        # Sample x_N-1, ..., x_1 by applying posterior
        for n_int in range(self.steps - 1, 0, -1):
            n = torch.full(
                (n_samples,), n_int, device=tmax.device, dtype=torch.long
            )
            x_n_1 = self.sample_posterior(x_n=x_n_1, n=n)

        # Sample x_0
        n = torch.full(
            (n_samples,), n_int - 1, device=tmax.device, dtype=torch.long
        )
        x_0, _, _, _ = self.sample_x_0(n=n, x_n=x_n_1)

        return x_0

    def sample_x_0(
        self, n: TensorType[int], x_n: Batch
    ) -> Tuple[Batch, Batch, Batch, Batch]:
        """
        Sample x_0 from x_n by classifying the intersection of x_0 and x_n and sampling from the intensity.

        Parameters
        ----------
        n : TensorType[int]
            Diffusion time steps
        x_n : Batch
            Batch of data

        Returns
        -------
        Tuple[Batch, Batch, Batch, Batch]
            x_0, classified_x_0, sampled_x_0, classified_not_x_0
        """
        (
            dif_time_emb,
            time_emb,
            event_emb,
        ) = self.compute_emb(n=n, x_n=x_n)

        # Sample x_0\x_n from intensity
        sampled_x_0 = self.intensity_model.sample(
            event_emb=event_emb,
            dif_time_emb=dif_time_emb,
            n_samples=1,
            x_n=x_n,
        )

        # Classify (x_0 ∩ x_n) from x_n
        x_n_and_x_0_logits = self.classifier_model(
            dif_time_emb=dif_time_emb, time_emb=time_emb, event_emb=event_emb
        )
        classified_x_0, classified_not_x_0 = x_n.thin(
            alpha=x_n_and_x_0_logits.sigmoid()
        )
        return (
            classified_x_0.add_events(sampled_x_0),
            classified_x_0,
            sampled_x_0,
            classified_not_x_0,
        )

    def sample_posterior(self, x_n: Batch, n: TensorType[int]) -> Batch:
        """
        Sample x_n-1 from x_n by predicting x_0 and then sampling from the posterior.

        Parameters
        ----------
        x_n : Batch
            Batch of data
        n : TensorType
            Diffusion time steps

        Returns
        -------
        Batch
            x_n-1
        """
        # Sample x_0 and x_n\x_0
        _, classified_x_0, sampled_x_0, classified_not_x_0 = self.sample_x_0(
            n=n, x_n=x_n
        )

        # Sample C
        x_0_kept, _ = sampled_x_0.thin(alpha=self.alpha_x0_kept[n - 1])

        # Sample D
        hpp = generate_hpp(
            tmax=x_n.tmax,
            n_sequences=x_n.batch_size,
            intensity=self.add_remove[n - 1],
        )

        # Sample E
        x_n_kept, _ = classified_not_x_0.thin(alpha=self.alpha_xn_kept[n - 1])

        # Superposition of B, C, D, E to attain x_n-1
        x_n_1 = (
            classified_x_0.add_events(hpp)
            .add_events(x_n_kept)
            .add_events(x_0_kept)
        )
        return x_n_1

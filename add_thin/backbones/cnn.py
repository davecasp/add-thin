import torch
import torch.nn as nn
import warnings

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


patch_typeguard()  # use before @typechecked


@typechecked
class CNNSeqEmb(nn.Module):
    """
    Dilated CNN with circular padding for sequence embedding.

    Parameters:
    ----------
    emb_layer : int
        Number of CNN layers
    input_dim : int
        Input dimension
    emb_dims : int
        Output dimension
    kernel_size : int, optional
        Convolution kernel size, by default 16
    """

    def __init__(
        self,
        emb_layer: int,
        input_dim: int,
        emb_dims: int,
        kernel_size: int = 16,
    ) -> None:
        super().__init__()
        # TODO: Change dilation values or make them configurable
        dilation = [1, 4, 8, 16, 32, 64]

        # Instantiate CNN layers
        layers = []
        for i in range(emb_layer):
            input_dim = input_dim if i == 0 else emb_dims
            layers.append(
                nn.Sequential(
                    *[
                        nn.Conv1d(
                            input_dim,
                            emb_dims,
                            kernel_size,
                            padding="same",
                            padding_mode="circular",
                            dilation=dilation[i],
                        ),
                        nn.GroupNorm(8, emb_dims),
                    ]
                )
            )
        self.activation = nn.ReLU(inplace=True)
        self.linear = nn.Linear(emb_dims, emb_dims)
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, x: TensorType[float, "batch", "sequence", "embedding"]
    ) -> TensorType[float, "batch", "sequence", "embedding"]:
        """
        Parameters:
        ----------
        x : TensorType[float, "batch", "sequence", "embedding"]
            Input tensor

        Returns:
        -------
        TensorType[float, "batch", "sequence", "embedding"]
            Sequence embedding
        """
        # Zero pad if sequence is shorter than 30, TODO: might want ot think about different solution
        if x.shape[1] < 30:
            # warnings.warn(
            #     f"Sequence is shorter than 30, padding with zeros. {x.shape}"
            # )
            x_before = x.shape[1]
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        (x.shape[0], 30 - x.shape[1], x.shape[2]),
                        device=x.device,
                    ),
                ],
                dim=1,
            )
        else:
            x_before = None

        # Swap axes to make sequence dimension the last dimension
        x = x.swapaxes(1, 2)

        # Apply residual CNN layers
        for l in self.layers:
            x = x + self.activation(l(x))

        # Swap axes back
        x = x.swapaxes(1, 2)

        # Remove padding
        if x_before is not None:
            x = x[:, :x_before]
        return self.linear(x)

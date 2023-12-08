import torch
import torch.nn as nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


@typechecked
class PointClassifier(nn.Module):
    """
    Classifier to predict the intersection of x_0 and x_n given x_n.

    Parameters:
    ----------

    hidden_dims : int
        Number of hidden dimensions
    layer : int
        Number of layers
    """

    def __init__(
        self,
        hidden_dims: int,
        layer: int,
    ) -> None:
        super().__init__()
        input_dim = 3 * hidden_dims

        # Instantiate MLP for the classifier
        layers = [nn.Linear(input_dim, hidden_dims), nn.ReLU()]
        for _ in range(layer - 1):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims, 1))
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        dif_time_emb: TensorType[float, "batch", "embedding"],
        time_emb: TensorType[float, "batch", "sequence", "time_emb"],
        event_emb: TensorType["batch", "sequence", "embedding"],
    ) -> TensorType[float, "batch", "sequence"]:
        """
        Parameters:
        ----------
        dif_time_emb : TensorType[float, "batch", "embedding"]
            Embedding of the diffusion time
        time_emb : TensorType[float, "batch", "sequence", "time_emb"]
            Embedding of the event times
        event_emb : TensorType["batch", "sequence", "embedding"]
            Context embedding of the events

        Returns:
        -------
        logits : TensorType[float, "batch", "sequence"]
            Logits for each event in the sequences
        """
        # Concatenate embeddings
        _, L, _ = time_emb.shape
        x = torch.cat(
            [
                time_emb,
                event_emb,
                dif_time_emb.unsqueeze(1).repeat(1, L, 1),
            ],
            dim=-1,
        )

        logits = self.model(x).squeeze(-1)
        return logits

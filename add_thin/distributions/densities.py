import torch
import torch.distributions as D


class Normal(D.Normal):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """
        Intantiate normal with specific mean and variance.

        Parameters:
        ----------
        mean : torch.Tensor
            Mean of the normal distribution.
        std : torch.Tensor
            Standard deviation of the normal distribution.
        """
        # TODO might want to change std and mean
        super().__init__(mean.sigmoid(), torch.exp(-torch.abs(std)))

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Change cdf to be on [0, x].

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Truncated CDF of the input tensor.
        """
        return super().cdf(x) - super().cdf(torch.zeros_like(x))


DISTRIBUTIONS = {"normal": Normal}

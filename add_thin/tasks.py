import pytorch_lightning as pl
import torch
import torch.nn as nn

from add_thin.data import Batch
from add_thin.metrics import (
    MMD,
    forecast_wasserstein,
    lengths_distribution_wasserstein_distance,
)


class Tasks(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay: float,
        weight_decay: float = 0.0,
        lr_schedule=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate

        self.model = model
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        Compute BCE loss for the classification task.
        """
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask.flatten()]
        target = x_n.kept.flatten()[x_n.mask.flatten()]
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        Compute the average (over batch) negative log-likelihood of the event sequences.
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        Compute the loss for the classification and intensity.
        """
        intensity = self.intensity_loss(log_prob_x_0) / self.model.n_max

        classification = (
            self.classification_loss(x_n_int_x_0, x_n) / self.model.n_max
        )
        loss = classification + intensity
        return loss, classification, intensity

    def step(self, batch, name):
        """
        Apply model to batch and compute loss.
        """
        # Forward pass
        x_n_int_x_0, log_prob_x_0, x_n = self.model.forward(batch)

        # Compute loss
        loss, classification, intensity = self.get_loss(
            log_prob_x_0, x_n_int_x_0, x_n
        )

        # Log loss
        self.log(
            f"{name}/loss",
            loss.detach().item(),
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/log-likelihood",
            intensity.detach().item(),
            batch_size=batch.batch_size,
        )
        if classification is not None:
            self.log(
                f"{name}/BCE",
                classification.detach().item(),
                batch_size=batch.batch_size,
            )
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=500, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train/loss",
            },
        }


class Forecasting(Tasks):
    def __init__(
        self,
        model,
        learning_rate,
        lr_decay,
        weight_decay,
        lr_schedule,
    ):
        super().__init__(
            model, learning_rate, lr_decay, weight_decay, lr_schedule
        )

    def set_history(self, batch):
        # Sample random start time for forecast window
        times = (
            torch.rand((len(batch),), device=batch.tmax.device)
            * (batch.tmax - 2 * self.model.forecast_window)
            + self.model.forecast_window
        )
        # Get history, future, and bounds of forecast window
        history, future, forecast_end, forecast_start = batch.split_time(
            times, times + self.model.forecast_window
        )
        self.model.set_history(history)
        return future, forecast_end, forecast_start

    def training_step(self, batch, batch_idx):
        future, _, forecast_start = self.set_history(batch)

        # rescale forecast to [0, T], same for inter-event times tau
        future.time = (
            (future.time - forecast_start[:, None]) / self.model.forecast_window
        ) * future.tmax
        future.tau = (future.tau / (self.model.forecast_window)) * future.tmax

        loss = self.step(future, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.global_step >= 1:
            futures = []
            samples = []
            maes = []
            # sample 5 forecast horizons per batch
            for _ in range(5):
                future, tmax, tmin = self.set_history(batch)
                sample = self.model.sample(len(future), tmax=future.tmax)
                # rescale and shift to right forecast window
                sample.time = (sample.time / future.tmax) * (tmax - tmin)[
                    :, None
                ] + tmin[:, None]
                samples = samples + sample.to_time_list()
                futures = futures + future.to_time_list()
                maes.append(
                    torch.abs(future.mask.sum(-1) - sample.mask.sum(-1))
                    / (future.mask.sum(-1) + 1)
                )

            wasserstein = forecast_wasserstein(
                samples,
                futures,
                batch.tmax.detach().cpu().item(),
            )

            self.log(
                "val/MAE_counts",
                torch.cat(maes).mean(),
                batch_size=batch.batch_size,
            )
            self.log(
                "val/forecast_wasserstein_distance",
                wasserstein,
                batch_size=batch.batch_size,
            )

    def test_step(self, batch, batch_idx):
        pass


class DensityEstimation(Tasks):
    def __init__(
        self, model, learning_rate, lr_decay, weight_decay, lr_schedule
    ):
        super().__init__(
            model, learning_rate, lr_decay, weight_decay, lr_schedule
        )

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.global_step >= 1:
                sample = self.model.sample(1000, tmax=batch.tmax).to_time_list()

                mmd = MMD(
                    sample,
                    batch.to_time_list(),
                    batch.tmax.detach().cpu().numpy(),
                )[0]
                wasserstein = lengths_distribution_wasserstein_distance(
                    sample,
                    batch.to_time_list(),
                    batch.tmax.detach().cpu().numpy(),
                    self.model.n_max,
                )
                self.log("val/sample_mmd", mmd, batch_size=batch.batch_size)
                self.log(
                    "val/sample_count_wasserstein",
                    wasserstein,
                    batch_size=batch.batch_size,
                )

    def test_step(self, batch, batch_idx):
        pass

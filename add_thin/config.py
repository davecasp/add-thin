from pathlib import Path
from omegaconf import DictConfig

from add_thin.data import DataModule
from add_thin.diffusion.model import AddThin
from add_thin.backbones.classifier import PointClassifier
from add_thin.distributions.intensities import MixtureIntensity
from add_thin.tasks import DensityEstimation, Forecasting


def instantiate_datamodule(config: DictConfig, task_name):
    return DataModule(
        Path(config.root),
        config.name,
        batch_size=config.batch_size,
        forecast=task_name == "forecast",
    )


def instantiate_model(config: DictConfig, datamodule) -> AddThin:
    classifier = PointClassifier(
        hidden_dims=config.hidden_dims,
        layer=config.classifier_layer,
    )
    intensity = MixtureIntensity(
        n_components=config.mix_components,
        embedding_size=2 * config.hidden_dims,
        distribution="normal",
    )

    model = AddThin(
        classifier_model=classifier,
        intensity_model=intensity,
        max_time=datamodule.dataset.tmax.item(),
        steps=config.steps,
        hidden_dims=config.hidden_dims,
        emb_dim=config.hidden_dims,
        encoder_layer=config.encoder_layer,
        n_max=datamodule.n_max,
        kernel_size=config.kernel_size,
        forecast=datamodule.forecast_horizon,
    )
    return model


def instantiate_task(config: DictConfig, model):
    if config.name == "density":
        return DensityEstimation(
            model,
            config.learning_rate,
            config.lr_decay,
            config.weight_decay,
            config.lr_schedule,
        )
    elif config.name == "forecast":
        return Forecasting(
            model,
            config.learning_rate,
            config.lr_decay,
            config.weight_decay,
            config.lr_schedule,
        )

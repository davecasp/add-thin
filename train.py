#!/usr/bin/env python
import faulthandler
import logging
import warnings

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from seml.hydra import seml_observe_hydra

from add_thin.config import (
    instantiate_datamodule,
    instantiate_model,
    instantiate_task,
)
from add_thin.utils import (
    WandbModelCheckpoint,
    WandbSummaries,
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)


def get_callbacks(config):
    monitor = {"monitor": config.task.metric, "mode": "min"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            filename="best",
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
    ]

    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    return callbacks


# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# Stop lightning from pestering us about things we already know
warnings.filterwarnings(
    "ignore",
    "There is a wandb run already in progress",
    module="pytorch_lightning.loggers.wandb",
)
warnings.filterwarnings(
    "ignore",
    "The dataloader, [^,]+, does not have many workers",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(
    filter_device_available
)
log = get_logger()


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
@seml_observe_hydra()
def main(config: DictConfig):
    rng = set_seed(config)
    torch.use_deterministic_algorithms(True)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)

    print_config(config)
    wandb.init(
        entity=config.entity,
        project=config.project,
        group=config.group,
        name=config.name,
        resume="allow",
        id=config.id,
        mode=config.mode,
        dir=config.run_dir,
    )

    OmegaConf.save(config, wandb.run.dir + "/config_hydra.yaml")
    log.info(wandb.run.dir)
    log.info("Loading data")
    datamodule = instantiate_datamodule(config.data, config.task.name)
    datamodule.prepare_data()

    log.info(config.data.name)

    log.info("Instantiating model")
    model = instantiate_model(config.model, datamodule)

    task = instantiate_task(config.task, model)

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    log.info("Loading checkpoint")
    callbacks = get_callbacks(config)

    log.info("Instantiating trainer")

    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    log.info(
        f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
    )

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()

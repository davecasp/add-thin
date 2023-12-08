import os
import torch
from omegaconf import OmegaConf

from add_thin.config import instantiate_datamodule, instantiate_model
from add_thin.tasks import DensityEstimation, Forecasting
from add_thin.utils.seed import set_seed


def get_task(path, density=True, data_root="/path/to/data"):
    """
    Load task and datamodule for a given run path.

    Parameters:
    -----------
        path (str): Path to the model directory.
        data_root (str): Path to the data directory.

    Returns:
    -----------
        task (Task): The task object.
        datamodule (DataModule): The datamodule object.
    """
    model_path = path + "/checkpoints/best.ckpt"
    with open(path + "/config_hydra.yaml", "r") as stream:
        config = OmegaConf.load(stream)

    # get config and set seed
    OmegaConf.resolve(config)
    _ = set_seed(config)

    # load data
    config.data.root = data_root + config.data.root
    datamodule = instantiate_datamodule(config.data, config.task.name)
    datamodule.prepare_data()

    # load model
    model = instantiate_model(config.model, datamodule)
    if density:
        task = DensityEstimation.load_from_checkpoint(model_path, model=model)
    else:
        task = Forecasting.load_from_checkpoint(model_path, model=model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device)
    task.eval()

    return task, datamodule


def get_run_data(run_id, project_dir):
    """
    Get run data from a given run id.

    Parameters:
    -----------
        run_id (str): The run id.
        project_dir (str): The project directory.

    Returns:
    -----------
        data_name (str): The name of the dataset.
        seed (int): The seed.
        path (str): The path to the run directory.
    """
    data_name, seed, path = None, None, None
    for _, dirnames, _ in os.walk(project_dir):
        for dirname in dirnames:
            if dirname.split("-")[-1] == run_id:
                path = str(project_dir) + "/" + dirname + "/files/"
                with open(path + "/config_hydra.yaml", "r") as stream:
                    config = OmegaConf.load(stream)
                data_name = config.data.name
                seed = config.seed

    if path is None:
        raise ValueError(f"Run id {run_id} not found in {project_dir}.")

    return data_name, seed, path

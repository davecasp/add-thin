# Add and Thin: Diffusion for Temporal Point Processes


<!-- A one line description of the project -->
This is the reference implementation of our NeurIPS 2023 paper [Add and Thin: Diffusion for Temporal Point Processes][paper].

</div>

## Citation
If you build upon this work, please cite our paper as follows:
```
@inproceedings{luedke2023add,
    title={Add and Thin: Diffusion for Temporal Point Processes},
    author={David L{\"u}dke and Marin Bilo{\v{s}} and Oleksandr Shchur and Marten Lienen and Stephan G{\"u}nnemann},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=tn9Dldam9L}
}
```


## Getting started
<!-- This section summarizes the basic requirements and the installation process to properly run and reproduce the code -->

### Basic requirements
<!-- List of basic requirements needed to properly run the code -->
The code has been tested on a cluster of Linux nodes using [SLURM][slurm-site].<br>
We _cannot guarantee_ the functioning of the code if the following requirements are _not_ met:


### Installation
<!-- List the steps needed to properly install and run the code -->
> To properly install and run our code we recommend using a virtual environment (e.g., created via [`pyenv-virtualenv`][pyenv-virtualenv-site] or [`conda`][conda-site]).

The entire installation process consists of 3 steps. You can skip step 0 at you own "risk".

#### (_Optional_) Step 0: create a virtual environment
In the following we show how to create the environment via [`pyenv`][pyenv-site] and [`pyenv-virtualenv`][pyenv-virtualenv-site].
The steps are the following:
- install [`pyenv`][pyenv-site] (if you don't have it yet) by following the [original guidelines][pyenv-install-site];
- install the correct Python version:
    ```sh
    pyenv install 3.10.4
    ```
- create a virtual environment with the correct version of Python:
    ```sh
    pyenv virtualenv 3.10.4 add_thin
    ```

#### Step 1: clone the repository, change into it and (_optional_) activate the environment
This step allows you to download the code in your machine, move into the correct directory and (_optional_) activate the correct environment.
The steps are the following:
- clone the repository:
    ```sh
    git clone https://github.com/davecasp/add-thin.git
    ```
- change into the repository:
    ```sh
    cd add-thin
    ```
- (_optional_) activate the environment (everytime you'll enter the folder, the environment will be automatically activated)
    ```sh
    pyenv local add_thin
    ```

#### Step 2: install the code as a local package
All the required packages are defined in the `pyproject.toml` file and can be easily installed via [`pip`][pip-site] as following:
```sh
pip install -e .
```


<!-- Python & libraries websites -->
[python-site]: https://www.python.org
[pytorch-site]: https://pytorch.org
[pytorch-install-site]: https://pytorch.org/get-started/locally/
[pyg-site]: https://pytorch-geometric.readthedocs.io/en/latest/index.html#
[pyg-install-site]: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
[lit-site]: https://www.pytorchlightning.ai
[slurm-site]: https://slurm.schedmd.com/documentation.html
[pyenv-virtualenv-site]: https://github.com/pyenv/pyenv-virtualenv
[pyenv-site]: https://github.com/pyenv/pyenv
[pyenv-install-site]: https://github.com/pyenv/pyenv#installation
[conda-site]: https://docs.conda.io/en/latest/
[pip-site]: https://pip.pypa.io/en/stable/
<!-- Internal references -->
[installation-guide-ref]: ./docs/installation.md
<!-- Other variables -->
[paper]: https://www.cs.cit.tum.de/daml/add-thin

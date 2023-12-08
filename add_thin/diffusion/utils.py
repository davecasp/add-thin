import torch


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Compute betas for a given alpha_t_bar function.

    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    We use it to implement the cosine beta schedule from https://arxiv.org/abs/2112.10741

    Parameters
    ----------
    num_diffusion_timesteps : int
        Number of diffusion timesteps
    alpha_bar : callable
        Function that returns the cumulative product of (1-beta) over time from
        t = [0,1]
    max_beta : float
        Maximum value for beta

    Returns
    -------
    betas : torch.Tensor
        Beta values for each timestep
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

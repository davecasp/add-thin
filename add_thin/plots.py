import matplotlib.pyplot as plt
import numpy as np
import wandb

from add_thin.data import Batch, Sequence


def to_counting_process(t, grid, mask):
    return (mask[None, ...] * (grid[:, None, None] > t[None, ...])).sum(-1)


def sample_plots(sample, real, task, density=True, tmax=None):
    if tmax is not None:
        real = Batch.from_sequence_list(
            [Sequence(time=seq, tmax=tmax) for seq in real]
        )
    else:
        tmax = real.tmax

    sample = Batch.from_sequence_list(
        [Sequence(time=seq, tmax=tmax) for seq in sample]
    )

    samples_data = sample.time.detach().cpu().numpy()
    samples_mask = sample.mask.detach().cpu().numpy()
    real_data = real.time.detach().cpu().numpy()
    real_mask = real.mask.detach().cpu().numpy()
    tmax = sample.tmax.cpu().item()
    grid = np.linspace(0, tmax, 200)

    max_range = 1.3 * np.sum(real_mask, axis=-1).max()
    min_range = 0.7 * np.sum(real_mask, axis=-1).min()

    samples_data_qq = to_counting_process(
        samples_data, grid, sample.mask.detach().cpu().numpy()
    )
    real_data_qq = to_counting_process(
        real_data, grid, real.mask.detach().cpu().numpy()
    )

    fig, ax = plt.subplots(
        1, 2, sharey=True, sharex=True, figsize=(10, 5), dpi=300
    )
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        ax[0].plot(
            grid,
            np.quantile(samples_data_qq, q, axis=-1),
            alpha=1 - abs(q - 0.5),
            color="C1",
        )
        ax[0].set_title("QQ-Plot samples")
        ax[1].plot(
            grid,
            np.quantile(real_data_qq, q, axis=-1),
            alpha=1 - abs(q - 0.5),
            color="C1",
        )
        ax[1].set_title("QQ-Plot real data")
    plot = wandb.Image(fig, mode="RGB")
    plt.close(fig)

    task.logger.log_metrics({f"val/qqplot": plot})

    fig, ax = plt.subplots(
        1, 1, sharey=True, sharex=True, figsize=(20, 5), dpi=300
    )
    ax.set_ylabel("Frequency")
    ax.hist(
        np.sum(samples_mask, axis=-1),
        bins=100,
        label="Sampled",
        density=True,
        range=(min_range, max_range),
    )
    ax.hist(
        np.sum(real_mask, axis=-1),
        bins=100,
        alpha=0.3,
        label="Real data",
        density=True,
        range=(min_range, max_range),
    )
    ax.set_xlabel("Sequence length")
    ax.set_title(f"Count histogram")

    plt.legend()
    plot = wandb.Image(fig, mode="RGB")
    plt.close(fig)

    task.logger.log_metrics({f"val/samples": plot})
    if density:
        fig, ax = plt.subplots(
            1, 5, sharey=True, sharex=True, figsize=(25, 5), dpi=300
        )
        min_length = min(len(samples_data), len(real_data))
        for i in range(5):
            for n in range(min_length):
                ax[i].hist(
                    real_data[n][real_mask[n]],
                    cumulative=True,
                    histtype="step",
                    alpha=0.1,
                    color="r",
                    bins=1000,
                )
            ax[i].hist(
                samples_data[i][samples_mask[i]],
                cumulative=True,
                label="sample",
                histtype="step",
                alpha=1,
                color="b",
                bins=1000,
            )
        plt.legend()
        plot = wandb.Image(fig, mode="RGB")
        plt.close(fig)
        task.logger.log_metrics({f"val/sampled_tpps": plot})
    else:
        fig, ax = plt.subplots(
            1, 5, sharey=True, sharex=True, figsize=(25, 5), dpi=300
        )
        min_length = min(len(samples_data), len(real_data))
        for i in range(5):
            ax[i].hist(
                real_data[i][real_mask[i]],
                cumulative=True,
                histtype="step",
                alpha=0.1,
                color="r",
                bins=1000,
            )
            ax[i].hist(
                samples_data[i][samples_mask[i]],
                cumulative=True,
                label="sample",
                histtype="step",
                alpha=1,
                color="b",
                bins=1000,
            )
        plt.legend()
        plot = wandb.Image(fig, mode="RGB")
        plt.close(fig)
        task.logger.log_metrics({f"val/forecasted_tpps": plot})

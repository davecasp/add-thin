import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

FORECAST_HORIZON = {
    "reddit_askscience_comments": 4,
    "reddit_politics_submissions": 4,
    "taxi": 4,
    "pubg": 5,
    "twitter": 4,
    "yelp_airport": 4,
    "yelp_mississauga": 4,
}

SYNTHETIC = {
    "hawkes1": "Hawkes1",
    "hawkes2": "Hawkes2",
    "self_correcting": "SC",
    "nonstationary_poisson": "IPP",
    "stationary_renewal": "RP",
    "nonstationary_renewal": "MRP",
}
REAL = {
    "pubg": "PUBG",
    "reddit_askscience_comments": "Reddit-C",
    "reddit_politics_submissions": "Reddit-S",
    "taxi": "Taxi",
    "twitter": "Twitter",
    "yelp_airport": "Yelp1",
    "yelp_mississauga": "Yelp2",
}


@typechecked
class Sequence:
    def __init__(
        self,
        time: np.ndarray | TensorType[float, "events"],
        tmax: Union[np.ndarray, TensorType[float], float],
        device: Union[torch.device, str] = "cpu",
        kept_points: Union[np.ndarray, TensorType, None] = None,
    ) -> None:
        super().__init__()
        if not isinstance(time, torch.Tensor):
            time = torch.as_tensor(time)

        if tmax is not None:
            if not isinstance(tmax, torch.Tensor):
                tmax = torch.as_tensor(tmax)

        if kept_points is not None:
            if not isinstance(kept_points, torch.Tensor):
                kept_points = torch.as_tensor(kept_points)
            kept_points = kept_points

        self.time = time
        self.tmax = tmax
        self.kept_points = kept_points

        self.device = device
        self.to(device)
        tau = torch.diff(
            self.time,
            prepend=torch.as_tensor([0.0], device=device),
            append=torch.as_tensor([self.tmax], device=device),
        )
        self.tau = tau

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Sequence":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self


@typechecked
class Batch:
    def __init__(
        self,
        mask: TensorType[bool, "batch", "sequence"],
        time: TensorType[float, "batch", "sequence"],
        tau: TensorType[float, "batch", "sequence"],
        tmax: TensorType,
        unpadded_length: TensorType[int, "batch"],
        kept: Union[TensorType, None] = None,
    ):
        super().__init__()
        self.time = time
        self.tau = tau
        self.tmax = tmax
        self.kept = kept

        # Padding and mask
        self.unpadded_length = unpadded_length
        self.mask = mask

        self._validate()

    @property
    def batch_size(self) -> int:
        return self.time.shape[0]

    @property
    def seq_len(self) -> int:
        return self.time.shape[1]

    def __len__(self):
        return self.batch_size

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Batch":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self

    @staticmethod
    def from_sequence_list(sequences: List[Sequence]) -> "Batch":
        """
        Create batch from list of sequences.
        """
        # Pad sequences for batching
        tmax = torch.cat(
            [sequence.tmax.unsqueeze(dim=0) for sequence in sequences], dim=0
        ).max()
        tau = pad([sequence.tau for sequence in sequences])
        time = pad(
            [sequence.time for sequence in sequences], length=tau.shape[-1]
        )
        device = tau.device

        sequence_length = torch.tensor(
            [len(sequence) for sequence in sequences], device=device
        )

        if sequences[0].kept_points != None:
            kept_points = pad(
                [sequence.kept_points for sequence in sequences],
                length=tau.shape[-1],
            )
        else:
            kept_points = None

        # Compute event mask for batching
        mask = (
            torch.arange(0, tau.shape[-1], device=device)[None, :]
            < sequence_length[:, None]
        )

        batch = Batch(
            mask=mask,
            time=time,
            tau=tau,
            tmax=tmax,
            unpadded_length=sequence_length,
            kept=kept_points,
        )
        return batch

    def add_events(self, other: "Batch") -> "Batch":
        """
        Add batch of events to sequences.

        Parameters:
        ----------
        other : Batch
            Batch of events to add.

        Returns:
        -------
        Batch
            Batch of events with added events.
        """
        assert len(other) == len(
            self
        ), "The number of sequences to add does not match the number of sequences in the batch."
        other = other.to(self.time.device)
        tmax = max(self.tmax, other.tmax)

        if self.kept is None:
            kept = torch.cat(
                [
                    torch.ones_like(self.time, dtype=bool),
                    torch.zeros_like(other.time, dtype=bool),
                ],
                dim=-1,
            )
        else:
            kept = torch.cat(
                [self.kept, torch.zeros_like(other.time, dtype=bool)],
                dim=-1,
            )

        return self.remove_unnescessary_padding(
            time=torch.cat([self.time, other.time], dim=-1),
            mask=torch.cat([self.mask, other.mask], dim=-1),
            kept=kept,
            tmax=tmax,
        )

    def to_time_list(self):
        time = []
        for i in range(len(self)):
            time.append(self.time[i][self.mask[i]].detach().cpu().numpy())
        return time

    def concat(self, *others):
        time = [self.time] + [o.time for o in others]
        mask = [self.mask] + [o.mask for o in others]
        return self.remove_unnescessary_padding(
            time=torch.cat(time, 0),
            mask=torch.cat(mask, 0),
            kept=None,
            tmax=self.tmax,
        )

    @staticmethod
    def sort_time(
        time, mask: TensorType[bool, "batch", "sequence"], kept, tmax
    ):
        """
        Sort events by time.

        Parameters:
        ----------
        time : TensorType[float, "batch", "sequence"]
            Tensor of event times.
        mask : TensorType[bool, "batch", "sequence"]
            Tensor of event masks.
        kept : TensorType[bool, "batch", "sequence"]
            Tensor indicating kept events.
        tmax : TensorType[float]
            Maximum time of the sequence.

        Returns:
        -------
        time : TensorType[float, "batch", "sequence"]
            Tensor of event times.
        mask : TensorType[bool, "batch", "sequence"]
            Tensor of event masks.
        kept : TensorType[bool, "batch", "sequence"]
            Tensor indicating kept events.
        """
        # Sort time and mask by time
        time[~mask] = 2 * tmax
        sort_idx = torch.argsort(time, dim=-1)
        mask = torch.take_along_dim(mask, sort_idx, dim=-1)
        time = torch.take_along_dim(time, sort_idx, dim=-1)
        if kept is not None:
            kept = torch.take_along_dim(kept, sort_idx, dim=-1)
        else:
            kept = None
        time = time * mask
        return time, mask, kept

    @staticmethod
    def remove_unnescessary_padding(
        time, mask: TensorType[bool, "batch", "sequence"], kept, tmax
    ):
        """
        Remove unnescessary padding from batch.

        Parameters:
        ----------
        time : TensorType[float, "batch", "sequence"]
            Tensor of event times.
        mask : TensorType[bool, "batch", "sequence"]
            Tensor of event masks.
        kept : TensorType[bool, "batch", "sequence"]
            Tensor indicating kept events.
        tmax : TensorType[float]
            Maximum time of the sequence.

        Returns:
        -------
        Batch
            Batch of events without unnescessary padding.
        """
        # Sort by time
        time, mask, kept = Batch.sort_time(time, mask, kept, tmax=tmax)

        # Reduce padding along sequence length
        max_length = max(mask.sum(-1)).int()
        mask = mask[:, : max_length + 1]
        time = time[:, : max_length + 1]
        if kept is not None:
            kept = kept[:, : max_length + 1]

        # compute interevent times
        time_tau = torch.where(mask, time, tmax)
        tau = torch.diff(
            time_tau, prepend=torch.zeros_like(time_tau)[:, :1], dim=-1
        )
        tau = tau * mask

        return Batch(
            mask=mask,
            time=time,
            tau=tau,
            tmax=tmax,
            unpadded_length=mask.sum(-1).long(),
            kept=kept,
        )

    def thin(self, alpha: TensorType[float]) -> Tuple["Batch", "Batch"]:
        """
        Thin events according to alpha.

        Parameters:
        ----------
        alpha : TensorType[float]
            Probability of keeping an event.

        Returns:
        -------
        keep : Batch
            Batch of kept events.
        remove : Batch
            Batch of removed events.
        """
        if alpha.dim() == 1:
            keep = torch.bernoulli(
                alpha.unsqueeze(1).repeat(1, self.seq_len)
            ).bool()
        elif alpha.dim() == 2:
            keep = torch.bernoulli(alpha).bool()
        else:
            raise Warning("alpha has too many dimensions")

        # remove from mask
        keep_mask = self.mask * keep
        rem_mask = self.mask * ~keep

        # shorten padding after removal
        return self.remove_unnescessary_padding(
            time=self.time * keep_mask,
            mask=keep_mask,
            kept=self.kept * keep_mask if self.kept is not None else self.kept,
            tmax=self.tmax,
        ), self.remove_unnescessary_padding(
            time=self.time * rem_mask,
            mask=rem_mask,
            kept=self.kept * rem_mask if self.kept is not None else self.kept,
            tmax=self.tmax,
        )

    def split_time(
        self,
        t_min: TensorType[float],
        t_max: TensorType[float],
    ) -> Tuple["Batch", "Batch", TensorType, TensorType]:
        """
        Split events according to time.

        Parameters:
        ----------
        t_min : TensorType[float]
            Minimum time of events to keep.
        t_max : TensorType[float]
            Maximum time of events to keep.

        Returns:
        -------
        history : Batch
            Batch of events before t_min.
        forecast : Batch
            Batch of events between t_min and t_max.
        t_max : TensorType
            Maximum time of events to keep.
        t_min : TensorType
            Minimum time of events to keep.
        """
        assert t_min.dim() == 1, "time has too many dimensions"
        assert t_max.dim() == 1, "time has too many dimensions"

        history_mask = self.time < t_min[:, None]
        forecast_mask = (self.time < t_max[:, None]) & ~history_mask

        # remove from mask
        forecast_mask = self.mask & forecast_mask
        history_mask = self.mask & history_mask

        # more than 5 events in history and more than one to be predicted
        batch_mask = (forecast_mask.sum(-1) > 1) & (history_mask.sum(-1) > 5)

        # shorten padding after removal
        return (
            self.remove_unnescessary_padding(
                time=(self.time * history_mask)[batch_mask],
                mask=history_mask[batch_mask],
                kept=None,
                tmax=self.tmax,
            ),
            self.remove_unnescessary_padding(
                time=(self.time * forecast_mask)[batch_mask],
                mask=forecast_mask[batch_mask],
                kept=None,
                tmax=self.tmax,
            ),
            t_max[batch_mask],
            t_min[batch_mask],
        )

    def _validate(self):
        """
        Validate batch, esp. masking.
        """
        # Check mask
        # mask as long as seq len;
        assert (self.mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
        assert (self.time * self.mask == self.time).all(), "wrong mask"

        assert torch.allclose(
            self.tau.cumsum(-1) * self.mask, self.time * self.mask, atol=1e-5
        ), "wrong tau"

        assert self.tau.shape == (
            self.batch_size,
            self.seq_len,
        ), f"tau has wrong shape {self.tau.shape}, expected {(self.batch_size, self.seq_len)}"


@typechecked
def pad(sequences, length: Union[int, None] = None, value: float = 0):
    """
    Utility function to generate padding and mask for sequences.
    Parameters:
    ----------
            sequences: List of sequences.
            value: float = 0,
            length: Optional[int] = None,
    Returns:
    ----------
            sequences: Padded sequence,
                shape (batch_size, seq_length)
            mask: Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len)
    """

    # Pad first sequence to enforce padding length
    if length:
        device = sequences[0].device
        dtype = sequences[0].dtype
        tensor_length = sequences[0].size(0)
        intial_pad = torch.empty(
            torch.Size([length]) + sequences[0].shape[1:],
            dtype=dtype,
            device=device,
        ).fill_(value)
        intial_pad[:tensor_length, ...] = sequences[0]
        sequences[0] = intial_pad

    sequences = pad_sequence(
        sequences, batch_first=True, padding_value=value
    )  # [order]

    return sequences


@typechecked
class SequenceDataset(torch.utils.data.Dataset):
    """Dataset of variable-length event sequences."""

    def __init__(
        self,
        sequences: List[Sequence],
    ):
        self.sequences = sequences
        self.tmax = sequences[0].tmax

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        return sequence

    def __len__(self) -> int:
        return len(self.sequences)

    def to(self, device: [torch.device, str]):
        for sequence in self.sequences:
            sequence.to(device)


@typechecked
class DataModule(pl.LightningDataModule):
    """
    Datamodule for variable length event sequences for temporal point processes.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.
    split_seed : int
        Seed for random split.
    batch_size : int
        Batch size.
    train_size : float
        Percentage of data to use for training.
    val_size : float
        Percentage of data to use for validation.
    forecast : bool
        Whether to use the dataset for forecasting.
    """

    def __init__(
        self,
        root: Path,
        name: str,
        split_seed: int = 80672983,
        batch_size: int = 32,
        train_size: float = 0.6,
        val_size: float = 0.2,
        forecast: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.train_percentage = train_size
        self.val_percentage = val_size
        self.name = name
        self.forecast = forecast

        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """Load sequence data from root."""
        time_sequences = load_sequences(self.root, self.name)

        if self.forecast:
            self.forecast_horizon = FORECAST_HORIZON[self.name]
        else:
            self.forecast_horizon = None

        self.dataset = SequenceDataset(sequences=time_sequences)
        self.tmax = self.dataset.tmax
        self.train_size = int(self.train_percentage * len(self.dataset))
        self.val_size = int(self.val_percentage * len(self.dataset))
        self.test_size = len(self.dataset) - (self.train_size + self.val_size)

        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [self.train_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

        self.get_statistics()

    def get_statistics(self):
        # Get train stats
        seq_lengths = []
        for i in range(len(self.train_data)):
            seq_lengths.append(len(self.train_data[i]))
        self.n_max = max(seq_lengths)

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=Batch.from_sequence_list,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=len(self.val_data),  # evaluate all at once
            collate_fn=Batch.from_sequence_list,
            num_workers=0,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=len(self.test_data),  # evaluate all at once
            collate_fn=Batch.from_sequence_list,
            num_workers=0,
            drop_last=False,
        )


def load_sequences(root, name: str) -> List[Sequence]:
    """Load dataset.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.

    Returns:
    -------
    time_sequences : List[Sequence]
        List of event sequences.

    """
    path = os.path.join(root, f"{name}.pkl")
    loader = torch.load(path, map_location=torch.device("cpu"))

    sequences = loader["sequences"]
    time = [seq["arrival_times"] for seq in sequences]
    tmax = loader["t_max"]

    time_sequences = [
        Sequence(sequence_time, tmax=tmax) for sequence_time in time
    ]

    return time_sequences

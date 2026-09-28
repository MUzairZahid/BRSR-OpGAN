"""Loading the BRSR benchmark (Zenodo, DOI 10.5281/zenodo.23010395).

Expected files (download them with ``python download_data.py``):

    data/brsr_{train,validation,test}.h5            BRSR (blind: AWGN + echo + CCI)
    data/awgn_baseline_{train,validation,test}.h5   AWGN-Baseline
    data/{brsr,awgn_baseline}_metadata.csv          per-sample metadata (optional)

Each HDF5 file holds ``clean`` and ``noisy`` [N, 2, 1024] float32 (channel 0 = I,
channel 1 = Q), ``label`` [N] (1..12), ``snr_db`` [N], ``gen_index`` [N] and, for
BRSR, ``distortions`` [N, 3, 2, 1024] (AWGN, echo, CCI components).
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATASETS = ("brsr", "awgn_baseline")
SPLITS = ("train", "validation", "test")
CLASS_NAMES = ["LFM", "Costas", "BPSK", "Frank", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]


def h5_path(data_dir, dataset, split):
    return os.path.join(data_dir, f"{dataset}_{split}.h5")


def load_split(data_dir, dataset, split, load_distortions=False):
    """Read one split into memory as numpy arrays."""
    path = h5_path(data_dir, dataset, split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python download_data.py --dataset {dataset}")
    with h5py.File(path, "r") as f:
        out = {k: f[k][:] for k in ("clean", "noisy", "label", "snr_db")}
        out["distortions"] = f["distortions"][:] if (load_distortions and "distortions" in f) else None
    return out


class RadarSignalDataset(Dataset):
    """Paired clean/noisy radar signals.

    With ``normalize=True`` every signal is min-max scaled to [-1, 1] per channel with
    its own minimum and maximum (the protocol of the BRSR-OpGAN paper). The per-channel
    min/max are returned so outputs can be mapped back to physical units.

    Each item: ((clean, clean_min, clean_max), (noisy, noisy_min, noisy_max), label, snr_db, distortions)
    """

    def __init__(self, clean, noisy, label, snr_db, distortions=None, normalize=True):
        self.clean = torch.as_tensor(clean, dtype=torch.float32)
        self.noisy = torch.as_tensor(noisy, dtype=torch.float32)
        self.label = torch.as_tensor(label, dtype=torch.long)
        self.snr_db = torch.as_tensor(snr_db, dtype=torch.float32)
        self.distortions = None if distortions is None else torch.as_tensor(distortions, dtype=torch.float32)
        self.normalize = normalize
        self.clean_min, self.clean_max = self.clean.amin(dim=2, keepdim=True), self.clean.amax(dim=2, keepdim=True)
        self.noisy_min, self.noisy_max = self.noisy.amin(dim=2, keepdim=True), self.noisy.amax(dim=2, keepdim=True)

    def __len__(self):
        return len(self.clean)

    @staticmethod
    def normalize_signal(signal, signal_min, signal_max):
        return 2 * (signal - signal_min) / (signal_max - signal_min) - 1

    def __getitem__(self, i):
        clean, noisy = self.clean[i], self.noisy[i]
        cmin, cmax, nmin, nmax = self.clean_min[i], self.clean_max[i], self.noisy_min[i], self.noisy_max[i]
        if self.normalize:
            clean = self.normalize_signal(clean, cmin, cmax)
            noisy = self.normalize_signal(noisy, nmin, nmax)
        dist = self.distortions[i] if self.distortions is not None else torch.empty(0)
        return (clean, cmin, cmax), (noisy, nmin, nmax), self.label[i], self.snr_db[i], dist


def denormalize_signal(normalized_signal, original_min, original_max):
    """Inverse of the [-1, 1] min-max scaling."""
    return (normalized_signal + 1) / 2 * (original_max - original_min) + original_min


def make_dataloaders(data_dir, dataset, batch_size, splits=SPLITS, normalize=True, num_workers=0,
                     pin_memory=False, load_distortions=False):
    loaders = {}
    for split in splits:
        d = load_split(data_dir, dataset, split, load_distortions)
        ds = RadarSignalDataset(d["clean"], d["noisy"], d["label"], d["snr_db"], d["distortions"], normalize)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=pin_memory)
    return loaders


def load_metadata(data_dir, dataset):
    """Per-sample metadata as a pandas DataFrame, or None if the CSV is not present."""
    path = os.path.join(data_dir, f"{dataset}_metadata.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    return pd.read_csv(path)

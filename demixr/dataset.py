import torch
import torchaudio
import os
import glob
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import load_audio

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2


class DemixrDataset(Dataset):
    def __init__(self, paths, instruments=["bass", "drums", "other", "vocals"]):
        self.paths = paths
        self.instruments = instruments

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        song = self.paths[idx]
        input, _ = load_audio(song["mixture"])

        target_stems = [load_audio(song[stem]) for stem in self.instruments]
        target = np.concatenate(target_stems, axis=0)

        return input, target


def create_dataloaders(dataset_paths, batch_size):
    dataset = DemixrDataset(dataset_paths["train"])
    test_dataset = DemixrDataset(dataset_paths["test"])

    lengths = [round(len(dataset) * split) for split in [TRAIN_SPLIT, VALIDATION_SPLIT]]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=lengths)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

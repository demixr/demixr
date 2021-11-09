import torch
import torchaudio
import os
import glob
import librosa
import h5py
from torch.utils.data import Dataset

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2


class DemixrDataset(Dataset):
    def __init__(
        self, dataset, hdf_dir, instruments=["bass", "drums", "other", "vocals"]
    ):
        self.dataset = dataset
        self.hdf_dir = hdf_dir
        self.instruments = instruments

    def __len__(self):
        return self.len

    def __build_path__(self, idx, label):
        return os.path.join(self.folders[idx], f"{label}.wav")

    def __getitem__(self, idx):
        x, _ = torchaudio.load(self.__build_path__(idx, self.input_file))
        y, _ = torchaudio.load(self.__build_path__(idx, self.output_file))
        return x, y


def create_dataloaders(musdbhq_dict):
    dataset = DemixrDataset(musdbhq_dict, "train")
    test_dataset = DemixrDataset(musdbhq_dict, "test")

    lengths = [round(len(dataset) * split) for split in [TRAIN_SPLIT, VALIDATION_SPLIT]]
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=lengths)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader

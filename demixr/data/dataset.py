import torch
import torchaudio
import os
import glob

import numpy as np

TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

class DemixrDataset(torch.utils.data.Dataset):
    def __init__(self, path, input_file='mixture', output_file='vocals'):
        self.path = path

        self.input_file = input_file
        self.output_file = output_file

        self.folders = sorted(glob.glob(os.path.join(self.path, '*')))
        self.len = len(self.folders)


    def __len__(self):
        return self.len


    def __build_path__(self, idx, label):
        return os.path.join(self.folders[idx], f'{label}.wav')


    def __getitem__(self, idx):
        x, _ = torchaudio.load(self.__build_path__(idx, self.input_file))
        y, _ = torchaudio.load(self.__build_path__(idx, self.output_file))
        return x, y


def create_dataloaders(data_path, output_file='vocals'):
    dataset = DemixrDataset(os.path.join(data_path, 'train'), output_file=output_file)
    test_dataset = DemixrDataset(os.path.join(data_path, 'test'), output_file=output_file)

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
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader

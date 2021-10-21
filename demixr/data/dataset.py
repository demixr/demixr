import torch
import torchaudio
import os
import glob

import numpy as np

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
import torch
import torchaudio
import os
import glob

import numpy as np

class DemixrDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.folders = sorted(glob.glob(os.path.join(self.path, '*')))
        self.len = len(self.folders)

        self.labels = np.asarray(['mixture', 'drums', 'bass', 'other', 'vocals'])


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        mixes = []
        for label in self.labels:
            mixes.append(torchaudio.load(os.path.join(self.folders[idx], f"{label}.wav")))
        print(mixes)
        return None
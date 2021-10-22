import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter


class demixr(nn.Module):

    def __init__(
            self,
            n_features=64,
            out_features=4096,
            n_layers=2,
            n_hidden=256,
            drop_prob=0.5
    ):
        super(demixr, self).__init__()

        self.n_features = n_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.drop_prob = drop_prob

        self.lstm = LSTM(self.n_features, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc1 = nn.Linear(20)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)

        return x




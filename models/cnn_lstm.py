import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
CNN-LSTM implementation, faithfully adapted from Elmadjian et al. (2023)
SOURCE: https://github.com/elmadjian/OEMC/blob/main/cnn_lstm.py

Matches the source exactly:
  - input_size = timesteps (Conv1d channel dim; features occupy the
    length dim) — same non-standard-but-intentional axis convention
    as tcn.py, consistent with the paper's seq2one design.
  - No transpose in forward() — input (batch, timesteps, features)
    fed directly into the conv stack.
  - 3 Conv1d layers (default filters 32/16/8), each followed by
    BatchNorm1d + ReLU; padding_mode='replicate' (NOT the PyTorch
    default 'zeros').
  - Dropout is inserted only BEFORE the 2nd and 3rd conv layers, not
    before the 1st — matches source exactly, not a uniform-dropout
    reimplementation.
  - Output taken from the LAST timestep of the LSTM sequence
    (out[:, -1, :]), consistent with the seq2one architecture.
'''


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout,
                 features, lstm_layers, conv_filters=(32, 16, 8),
                 bidirectional=False):
        """
        input_size    : number of timesteps — Conv1d channel dim
                        (matches source: TCN/CNN_LSTM(args.timesteps, ...))
        output_size   : number of classes
        kernel_size   : conv kernel size (paper default: 5)
        dropout       : dropout rate (paper default: 0.25), applied
                        before the 2nd and 3rd conv layers only
        features      : number of input features — LSTM input_size
                        (i.e. X.shape[1] from the feature tensor)
        lstm_layers   : number of stacked LSTM layers (paper: 2)
        conv_filters  : filter sizes per conv layer (paper default:
                        (32, 16, 8))
        bidirectional : kept for API parity with the source file, where
                        CNN_LSTM(..., bidirectional=True) was one way
                        the original codebase constructed the BiLSTM
                        variant. This project uses the separate
                        CNN_BiLSTM class instead (cnn_bilstm.py) for
                        that purpose; leave this False for CNN_LSTM.
        """
        super(CNN_LSTM, self).__init__()
        self.conv_filters = conv_filters
        self.lstm_layers  = lstm_layers

        conv_layers = []
        padding = int(np.floor((kernel_size - 1) / 2))
        for i, filt in enumerate(self.conv_filters):
            input_conv = input_size if i == 0 else conv_filters[i - 1]
            if i > 0:
                conv_layers += [nn.Dropout(dropout)]
            conv_layers += [nn.Conv1d(input_conv, conv_filters[i], kernel_size,
                                       padding=padding, padding_mode='replicate')]
            conv_layers += [nn.BatchNorm1d(conv_filters[i])]
            conv_layers += [nn.ReLU()]
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = TimeDistributed(nn.Flatten(), batch_first=True)

        hidden_state = 32
        if bidirectional:
            hidden_state = 16
        self.lstm = nn.LSTM(input_size=features, bidirectional=bidirectional,
                             hidden_size=hidden_state, num_layers=lstm_layers,
                             batch_first=True)

        linear = nn.Linear(32, output_size)
        linear.weight.data.normal_(0, 0.01)
        self.output = TimeDistributed(linear, batch_first=True)

    def forward(self, x):
        # x: (batch, timesteps, features) — no transpose, matches source.
        # Conv1d receives (batch, timesteps, features):
        #   channel dim = timesteps, length dim = features
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.flatten(out)
        out, _ = self.lstm(out)
        out = self.output(out[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out

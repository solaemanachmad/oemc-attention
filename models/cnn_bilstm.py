import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
CNN-BiLSTM implementation, faithfully adapted from Elmadjian et al. (2023)
SOURCE: https://github.com/elmadjian/OEMC/blob/main/cnn_blstm.py

Matches the source exactly:
  - input_size = timesteps (Conv1d channel dim; features occupy the
    length dim) — same convention as tcn.py / cnn_lstm.py.
  - No transpose in forward() — input (batch, timesteps, features)
    fed directly into the conv stack.
  - 3 Conv1d layers (default filters 32/16/8), each followed by
    BatchNorm1d + ReLU; padding_mode='replicate'.
  - Dropout inserted only BEFORE the 2nd and 3rd conv layers.
  - Bidirectional LSTM with hidden_size=16 per direction (so the
    concatenated output is 32-dim, matching CNN_LSTM's single-direction
    hidden_size=32 — this keeps the final Linear(32, output_size) layer
    identical in shape between the two baselines).
  - NOTE: unlike cnn_lstm.py's TimeDistributed(..., batch_first=True),
    the source cnn_blstm.py constructs its TimeDistributed layers and
    nn.LSTM WITHOUT batch_first=True. This is intentional and matches
    the original repository exactly — do not "fix" this to
    batch_first=True, as that would change the tensor layout the
    source code relies on (see forward() below).
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


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout,
                 features, blstm_layers, conv_filters=(32, 16, 8)):
        """
        input_size    : number of timesteps — Conv1d channel dim
        output_size   : number of classes
        kernel_size   : conv kernel size (paper default: 5)
        dropout       : dropout rate (paper default: 0.25), applied
                        before the 2nd and 3rd conv layers only
        features      : number of input features — BiLSTM input_size
                        (i.e. X.shape[1] from the feature tensor)
        blstm_layers  : number of stacked BiLSTM layers (paper: 2)
        conv_filters  : filter sizes per conv layer (paper default:
                        (32, 16, 8))
        """
        super(CNN_BiLSTM, self).__init__()
        self.conv_filters = conv_filters

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

        # NOTE: batch_first left at default (False) — matches source exactly.
        self.flatten = TimeDistributed(nn.Flatten())
        self.blstm   = nn.LSTM(input_size=features, bidirectional=True,
                                hidden_size=16, num_layers=blstm_layers)

        linear = nn.Linear(32, output_size)
        linear.weight.data.normal_(0, 0.01)
        self.output = TimeDistributed(linear)

    def forward(self, x):
        # x: (batch, timesteps, features) — no transpose, matches source.
        # Conv1d receives (batch, timesteps, features):
        #   channel dim = timesteps, length dim = features
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.flatten(out)
        out, _ = self.blstm(out)
        out = self.output(out[:, -1, :])
        out = F.log_softmax(out, dim=1)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import TimeDistributed

class CNN(nn.Module):
    def __init__(self, input_size, output_size=4, conv_filters=[32, 32, 32], kernel_size=5, dense_units=[128, 64], dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        in_channels = input_size
        for out_channels in conv_filters:
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.dropout = nn.Dropout(dropout_rate)
        self.td_dense_layers = nn.ModuleList()
        prev_dim = conv_filters[-1]
        for dim in dense_units:
            self.td_dense_layers.append(TimeDistributed(nn.Linear(prev_dim, dim)))
            prev_dim = dim
        self.td_output = TimeDistributed(nn.Linear(prev_dim, output_size))

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = bn(F.relu(conv(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        for td_dense in self.td_dense_layers:
            x = F.relu(td_dense(x))
        x = self.td_output(x)
        x = F.log_softmax(x[:, -1, :], dim=1)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, output_size=4, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.time_distributed_dense1 = nn.Linear(hidden_size * 2, 100)
        self.time_distributed_dense2 = nn.Linear(100, output_size)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        dropped = self.dropout(lstm_out)
        td_out1 = F.relu(self.time_distributed_dense1(dropped))
        logits = self.time_distributed_dense2(td_out1)
        last_logits = logits[:, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        return log_probs

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, output_size=4, conv_filters=(32, 16, 8, 4), kernel_size=3, dropout_rate=0.3, dense_units=32, blstm_units=16, padding_mode='same', no_bidirectional=False):
        super(CNN_BiLSTM, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = input_size
        for i, out_channels in enumerate(conv_filters):
            padding = kernel_size // 2 if padding_mode == 'same' else 0
            layer = [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ]
            if i > 0:
                layer.append(nn.Dropout(dropout_rate))
            self.conv_layers.append(nn.Sequential(*layer))
            in_channels = out_channels
        self.fc_before_lstm = nn.Sequential(
            nn.Linear(in_channels, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.blstm = nn.LSTM(input_size=dense_units, hidden_size=blstm_units, num_layers=1, batch_first=True, dropout=0, bidirectional=not no_bidirectional)
        lstm_output_dim = blstm_units * (2 if not no_bidirectional else 1)
        self.out_layer = nn.Linear(lstm_output_dim, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)
        batch_size, seq_len, feat_dim = x.shape
        x = x.reshape(-1, feat_dim)
        x = self.fc_before_lstm(x)
        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.blstm(x)
        x = lstm_out[:, -1, :]
        x = torch.tanh(x)
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size=4, conv_filters=(32, 64), kernel_sizes=(5, 3), lstm_units=64, dropout_rate=0.3):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.relu2 = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_filters[1], hidden_size=lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        logits = self.fc(last_timestep)
        return F.log_softmax(logits, dim=1)

class CNN_Transformer(nn.Module):
    def __init__(self, input_size, output_size, filters=[208, 128, 192], kernels=[5, 3, 3], dense_units=208, transformer_blocks=10, head_size=216, num_heads=8, ff_dim=512, mlp_units=208, dropout=0.3, mlp_dropout=0.4, timesteps=5):
        super(CNN_Transformer, self).__init__()
        self.timesteps = timesteps
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, filters[0], kernel_size=kernels[0], padding='same'),
            nn.BatchNorm1d(filters[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], padding='same'),
            nn.BatchNorm1d(filters[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], padding='same'),
            nn.BatchNorm1d(filters[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, timesteps, filters[2]))
        encoder_layer = nn.TransformerEncoderLayer(d_model=filters[2], nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters[2], mlp_units),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_units, output_size),
            nn.LogSoftmax(dim=1) 
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)          
        x = self.conv_layers(x)         
        x = x.permute(0, 2, 1)          
        pos = self.pos_encoding[:, :x.size(1), :]
        x = x + pos                     
        x = self.transformer_encoder(x)  
        x = x.permute(0, 2, 1)          
        x = self.pool(x)                
        x = self.classifier(x)          
        return x
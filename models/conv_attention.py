import torch
import torch.nn as nn

class Conv_Attention(nn.Module):
    def __init__(self, input_size, d_model=256, output_size=4, dropout=0.3, timesteps=5, kernel_size=3, num_heads=4):
        super(Conv_Attention, self).__init__()
        padding = kernel_size // 2 

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = nn.Parameter(torch.randn(1, timesteps, d_model))

        self.encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = x + self.positional_encoding[:, :x.size(1), :]

        attn_output, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_output)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=2))
        x = self.classifier(x)
        return self.softmax(x)
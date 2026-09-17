import torch
import torch.nn as nn


class Conv_Attention(nn.Module):
    def __init__(self, input_size, d_model=256, output_size=4, dropout=0.3,
                 timesteps=5, kernel_size=3, num_heads=4,
                 use_attention=True, use_positional_encoding=True,
                 encoder_layers=3):
        """
        Architecture ablation flags (all default to the original,
        published configuration — no behaviour change unless a flag is
        explicitly overridden):

        use_attention (bool):
            If False, the multi-head self-attention block is skipped
            entirely. The attn_norm LayerNorm is still applied (without
            the residual attention term) so the ablation isolates the
            effect of attention itself, not the normalization.

        use_positional_encoding (bool):
            If False, the learnable positional_encoding is not added
            to the conv-embedded sequence before attention/encoder.

        encoder_layers (int, 0-3):
            Number of stacked Conv1d+ReLU refinement blocks after the
            attention stage. 0 = encoder is skipped (Identity),
            3 = original architecture (default).
        """
        super(Conv_Attention, self).__init__()
        if not (0 <= encoder_layers <= 3):
            raise ValueError(
                f"encoder_layers must be between 0 and 3, got {encoder_layers}"
            )

        padding = kernel_size // 2

        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        self.encoder_layers = encoder_layers

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if self.use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, timesteps, d_model)
            )

        # Conv refinement encoder — built dynamically so encoder_layers
        # can be 0 (Identity), 1, 2, or 3 (original) without needing a
        # separate model class per variant.
        if encoder_layers == 0:
            self.encoder = nn.Identity()
        else:
            layers = []
            for _ in range(encoder_layers):
                layers.append(nn.Conv1d(d_model, d_model,
                                         kernel_size=kernel_size, padding=padding))
                layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*layers)

        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads,
                dropout=dropout, batch_first=True
            )
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

        if self.use_positional_encoding:
            x = x + self.positional_encoding[:, :x.size(1), :]

        if self.use_attention:
            attn_output, _ = self.attention(x, x, x)
            x = self.attn_norm(x + attn_output)
        else:
            # No attention mixing — still normalize so the ablation
            # isolates the effect of attention itself, not LayerNorm.
            x = self.attn_norm(x)

        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=2))
        x = self.classifier(x)
        return self.softmax(x)

# import torch
# import torch.nn as nn

# class Conv_Attention(nn.Module):
#     def __init__(self, input_size, d_model=256, output_size=4, dropout=0.3, timesteps=5, kernel_size=3, num_heads=4):
#         super(Conv_Attention, self).__init__()
#         padding = kernel_size // 2 

#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=d_model, kernel_size=kernel_size, padding=padding)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.positional_encoding = nn.Parameter(torch.randn(1, timesteps, d_model))

#         self.encoder = nn.Sequential(
#             nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(),
#             nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
#             nn.ReLU(),
#             nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
#             nn.ReLU()
#         )
#         self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.attn_norm = nn.LayerNorm(d_model)
#         self.norm = nn.LayerNorm(d_model)

#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model // 2, output_size)
#         )
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = x.transpose(1, 2)
#         x = x + self.positional_encoding[:, :x.size(1), :]

#         attn_output, _ = self.attention(x, x, x)
#         x = self.attn_norm(x + attn_output)

#         x = x.transpose(1, 2)
#         x = self.encoder(x)
#         x = self.norm(x.mean(dim=2))
#         x = self.classifier(x)
#         return self.softmax(x)
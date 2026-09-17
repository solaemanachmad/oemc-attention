import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Skip-AttSeqNet — best-effort reimplementation from the architecture
description and Table 3 of:

  Wang, X., Fan, L., Li, H., Bi, X., Jiang, W., & Ma, X. (2025).
  "Skip-AttSeqNet: Leveraging skip connection and attention-driven
  Seq2seq model to enhance eye movement event detection in
  Parkinson's disease." Biomedical Signal Processing and Control,
  99, 106862. https://doi.org/10.1016/j.bspc.2024.106862

NO PUBLIC CODE IS AVAILABLE for this model (unlike TCN/CNN-LSTM/
CNN-BiLSTM, which are faithfully reimplemented from
github.com/elmadjian/OEMC). This implementation is instead derived
from the paper's Table 3 (architecture/hyperparameters), Eq. (3)-(8)
(encoder, skip connection, attention), and Section 2.2 prose. Several
details are NOT fully specified in the paper and are approximated
here — see inline notes. Report this model's results as a
"best-effort reimplementation", not a faithful reproduction.

Architecture (Table 3):
  Encoder : 4x Conv1D, filters (32,16,8,8), kernel=3, VALID padding,
            no pooling, stride 1, BatchNorm, dropout 0.2, ReLU.
            (Eq. 3: kernel slides over the TIME axis — standard
            Conv1D orientation, channels=features, length=timesteps.
            This differs from this project's TCN/CNN-LSTM/CNN-BiLSTM,
            which use Elmadjian et al.'s non-standard channel=timesteps
            convention — Skip-AttSeqNet is a separate paper/codebase
            and is NOT expected to follow that convention.)
  Skip    : Eq. (5) — concatenate the original input features with
            the encoder output along the feature dimension. Because
            VALID convolution shrinks the time dimension (by
            kernel_size-1 per layer, 4 layers => 8 total), the
            original input is center-cropped to match the encoder
            output's (shorter) time length before concatenation.
            The paper does not specify the exact crop alignment;
            center-cropping is the most natural choice and is used
            here (APPROXIMATION).
  Decoder : 2-layer BLSTM, hidden size 16 per direction (32 total),
            dropout 0.3.
  Attention: Eq. (6)-(8) — a single learnable query vector scores
            every decoder timestep via dot product, normalized with
            softmax, then used to compute a weighted sum over time
            (attention pooling to a single context vector). The paper
            does not give exact dimensions for the "spatial attention
            matrix" w; here w has the same dimensionality as the
            decoder's hidden state (APPROXIMATION).
  Classifier: 1 FC layer + softmax. This project uses NLLLoss
            throughout (see train.py's _make_criterion), so the
            output here is LogSoftmax rather than the paper's plain
            softmax — mathematically equivalent when paired with
            NLLLoss instead of categorical cross-entropy.

Hyperparameters (Table 3): Adam, lr=0.001, batch_size=5000,
epochs=1000, dropout 0.2 (CNN) / 0.3 (RNN). This project's shared
training protocol (300 epochs with early stopping, batch 2048)
is used instead for fair comparison across all models — see
Section 3.4 discussion in the manuscript.
'''


class SkipAttSeqNet(nn.Module):
    def __init__(self, input_size, output_size, features,
                 conv_filters=(32, 16, 8, 8), kernel_size=3,
                 cnn_dropout=0.2, rnn_hidden=16, rnn_layers=2,
                 rnn_dropout=0.3):
        """
        input_size   : number of timesteps (sequence length fed to the
                       model) — kept as the first positional argument
                       only for naming consistency with this project's
                       other model classes; used here purely for
                       validating the minimum sequence length needed
                       to survive 4 VALID convolutions.
        output_size  : number of classes
        features     : number of input features per timestep (paper:
                       10 = 5 temporal scales x {speed, direction});
                       this project's own preprocessed feature count
                       is used instead, whatever that is.
        conv_filters : encoder filter sizes (paper default: (32,16,8,8))
        kernel_size  : conv kernel size (paper default: 3)
        cnn_dropout  : dropout after each conv block (paper default: 0.2)
        rnn_hidden   : BLSTM hidden size PER DIRECTION (paper default: 16,
                       i.e. 32-dim after concatenating both directions)
        rnn_layers   : number of stacked BLSTM layers (paper default: 2)
        rnn_dropout  : dropout on the BLSTM (paper default: 0.3)
        """
        super(SkipAttSeqNet, self).__init__()

        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.time_reduction = len(conv_filters) * (kernel_size - 1)

        if input_size <= self.time_reduction:
            raise ValueError(
                f"timesteps={input_size} is too short for {len(conv_filters)} "
                f"VALID conv layers with kernel_size={kernel_size} (needs > "
                f"{self.time_reduction} timesteps, since each layer removes "
                f"{kernel_size - 1} steps). Increase --timesteps or reduce "
                f"conv_filters length."
            )

        # ---- Encoder: 4x Conv1D, VALID padding, BatchNorm, ReLU, dropout ----
        encoder_layers = []
        in_ch = features
        for out_ch in conv_filters:
            encoder_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=0),  # VALID
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(cnn_dropout),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Skip connection: concat(original_input, encoder_output) ----
        skip_dim = features + conv_filters[-1]

        # ---- Decoder: 2-layer BLSTM ----
        self.blstm = nn.LSTM(
            input_size=skip_dim, hidden_size=rnn_hidden,
            num_layers=rnn_layers, batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )
        decoder_dim = rnn_hidden * 2  # bidirectional

        # ---- Attention: learnable query, dot-product score, softmax, ----
        # ---- weighted sum over time (Eq. 6-8)                        ----
        self.attn_query = nn.Parameter(torch.randn(decoder_dim) * 0.01)

        # ---- Classifier: 1 FC layer ----
        self.fc = nn.Linear(decoder_dim, output_size)

    def _center_crop_time(self, x, target_len):
        """x: (batch, T, features) -> (batch, target_len, features),
        cropping equally from both ends of the time dimension."""
        T = x.size(1)
        excess = T - target_len
        start = excess // 2
        return x[:, start:start + target_len, :]

    def forward(self, x):
        # x: (batch, timesteps, features)
        original = x

        # Encoder expects (batch, channels=features, length=timesteps)
        enc_in = x.transpose(1, 2)
        enc_out = self.encoder(enc_in)              # (batch, C_out, T')
        enc_out = enc_out.transpose(1, 2)            # (batch, T', C_out)

        # Skip connection: crop original input to T', concat on feature dim
        T_prime = enc_out.size(1)
        cropped_original = self._center_crop_time(original, T_prime)
        skip_features = torch.cat([cropped_original, enc_out], dim=-1)
        # skip_features: (batch, T', features + C_out)   -- Eq. (5)

        # Decoder
        decoder_out, _ = self.blstm(skip_features)   # (batch, T', 2*rnn_hidden)

        # Attention pooling over time (Eq. 6-8)
        scores = torch.matmul(decoder_out, self.attn_query)   # (batch, T')
        attn_weights = F.softmax(scores, dim=1)                # (batch, T')
        context = torch.sum(
            decoder_out * attn_weights.unsqueeze(-1), dim=1
        )                                                       # (batch, 2*rnn_hidden)

        # Classifier
        logits = self.fc(context)
        return F.log_softmax(logits, dim=1)

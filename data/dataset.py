import torch
import numpy as np
from torch.utils.data import Dataset


class EM_Loader_Lookahead(Dataset):
    """
    Lookahead windowing strategy.

    Window direction : forward — [i : i+timesteps]
    Label assignment : Y[i + timesteps - 1]  (last timestep of the window)

    This is the conventional sliding-window approach for sequence
    classification: the model sees a window of past-to-present samples
    and predicts the label at the END of that window.

    Compatible with all model types.
    For TCN (conventional input mode): prepare_input() will transpose to
    (batch, features, timesteps) before feeding the model.
    """

    def __init__(self, X, Y, timesteps=1, stride=1):
        self.X = X
        self.Y = Y
        self.timesteps = timesteps
        self.stride = stride

        # All valid start indices where a full window fits
        self.indices = list(
            range(0, len(self.Y) - self.timesteps + 1, self.stride)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.timesteps
        x_seq = torch.tensor(self.X[start:end], dtype=torch.float)  # (timesteps, features)
        y     = torch.tensor(self.Y[end - 1],   dtype=torch.long)   # label = last timestep
        return x_seq, y


class EM_Loader_Lookback(Dataset):
    """
    Lookback windowing strategy.

    Window direction : backward — [i-timesteps : i]
    Label assignment : Y[i-1]  (the sample immediately before the window end)

    This mirrors the create_batches() behaviour in the Bai et al. (2018)
    reference implementation:

        b_X = [X[i-timesteps:i, :] for i in range(start, end)]
        b_Y = Y[start-1:end-1]

    The model looks back over the most recent 'timesteps' samples and
    predicts the label of the current position (before the window closes).

    Compatible with all model types.
    When used with TCN in 'paper' input mode, no transpose is applied and
    input_size is set to timesteps rather than features.
    """

    def __init__(self, X, Y, timesteps=1, stride=1):
        self.X = X
        self.Y = Y
        self.timesteps = timesteps
        self.stride = stride

        # Start from 'timesteps' so X[i-timesteps:i] is always valid
        self.indices = list(
            range(timesteps, len(self.Y), self.stride)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i     = self.indices[idx]
        x_seq = torch.tensor(self.X[i - self.timesteps:i], dtype=torch.float)  # (timesteps, features)
        y     = torch.tensor(self.Y[i - 1],                dtype=torch.long)   # label = Y[i-1]
        return x_seq, y


# ---------------------------------------------------------------------------
# Quick-reference
# ---------------------------------------------------------------------------
#
# loader_mode="lookahead"  →  EM_Loader_Lookahead
#   window : X[i : i+timesteps]
#   label  : Y[i + timesteps - 1]
#   use    : default for all models
#
# loader_mode="lookback"   →  EM_Loader_Lookback
#   window : X[i-timesteps : i]
#   label  : Y[i-1]
#   use    : TCN paper replication, or any model where the label
#            should correspond to the start of the NEXT window
# ---------------------------------------------------------------------------

LOADER_REGISTRY = {
    "lookahead": EM_Loader_Lookahead,
    "lookback":  EM_Loader_Lookback,
}
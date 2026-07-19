import torch
import torch.nn as nn
from torchinfo import summary

from .conv_attention import Conv_Attention
from .tcn import TCN
from .baselines import CNN, BiLSTM, CNN_LSTM, CNN_BiLSTM, CNN_Transformer


# ------------------------------------------------------------------ #
# Model registry
# ------------------------------------------------------------------ #

def get_model(model_type, model_params):
    registry = {
        "conv_attention":  Conv_Attention,
        "cnn":             CNN,
        "bilstm":          BiLSTM,
        "cnn_lstm":        CNN_LSTM,
        "cnn_bilstm":      CNN_BiLSTM,
        "tcn":             TCN,
        "cnn_transformer": CNN_Transformer,
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model type: '{model_type}'. "
                         f"Available: {list(registry.keys())}")
    return registry[model_type](**model_params)


# ------------------------------------------------------------------ #
# Weight initialisation helper (used by CNN_BiLSTM)
# ------------------------------------------------------------------ #

def init_weights_normal(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ------------------------------------------------------------------ #
# Model summary
#
# All models receive (batch, timesteps, features) — no special casing
# needed here. Each model handles its own internal reshaping if required
# (e.g. TCN transposes inside forward()).
# ------------------------------------------------------------------ #

def print_summary(model, model_type, input_size, timesteps):
    original_device = next(model.parameters()).device
    model = model.cpu()

    # Uniform input shape for all models: (batch, timesteps, features)
    dummy_input = torch.randn(1, timesteps, input_size)

    try:
        s = summary(
            model,
            input_data=(dummy_input,),
            col_names=["input_size", "output_size", "num_params"],
            depth=4,
            verbose=0,
        )
        print(s)
    except Exception as e:
        print(f"Failed to print model summary: {e}")
    finally:
        model = model.to(original_device)
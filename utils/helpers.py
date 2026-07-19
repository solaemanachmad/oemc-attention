import multiprocessing
import os
import json
import psutil
import torch
import random
import numpy as np
import torch.nn as nn
import wandb

from utils.logger import logger

_DEVICE_LOGGED = False


def get_device():
    global _DEVICE_LOGGED
    device = None
    is_main_process = multiprocessing.current_process().name == 'MainProcess'

    if torch.cuda.is_available():
        if not _DEVICE_LOGGED and is_main_process:
            logger.info("Using NVIDIA CUDA (or AMD ROCm)")
        device = torch.device('cuda')

    if device is None:
        try:
            import torch_directml
            if torch_directml.is_available():
                if not _DEVICE_LOGGED and is_main_process:
                    logger.info("Using GPU: AMD Radeon via DirectML")
                device = torch_directml.device()
        except ImportError:
            pass

    if device is None and hasattr(torch.backends, 'mps') \
            and torch.backends.mps.is_available():
        if not _DEVICE_LOGGED and is_main_process:
            logger.info("Using GPU: Apple Silicon (MPS)")
        device = torch.device('mps')

    if device is None:
        if not _DEVICE_LOGGED and is_main_process:
            logger.warning("No GPU detected. Using CPU. Training may be slow.")
        device = torch.device('cpu')

    _DEVICE_LOGGED = True
    return device


# ------------------------------------------------------------------ #
# TimeDistributed
# ------------------------------------------------------------------ #

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
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
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y


# ------------------------------------------------------------------ #
# EarlyStopping
# Pass -val_loss to stop on loss plateau (matches original working code).
# ------------------------------------------------------------------ #

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter    = 0


# ------------------------------------------------------------------ #
# Randomness
# ------------------------------------------------------------------ #

def set_randomness(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
# Naming helpers
# ------------------------------------------------------------------ #

def _lr_str(lr):
    """Format lr for filename: 0.001 -> 001, 0.01 -> 01, 0.0001 -> 0001"""
    return f"{lr:.6f}".replace("0.", "").rstrip("0") or "0"


def _do_str(dropout):
    """Format dropout for filename: 0.2 -> do02, 0.25 -> do25, 0.3 -> do03"""
    frac = f"{dropout:.2f}".split(".")[-1]  # "20", "25", "30"
    return frac.rstrip("0").zfill(2) or "00"  # "02", "25", "03"


def build_config_tag(model_type=None, timesteps=None,
                     # conv_attention
                     num_heads=None, d_model=None, kernel_size=None,
                     # tcn
                     tcn_channel_size=None, tcn_num_levels=None,
                     # cnn_lstm / cnn_bilstm
                     lstm_layers=None, conv_filters=None,
                     # shared
                     dropout=None, lr=None, batch_size=None):
    """
    Build a model-specific compact config string for file naming.

    conv_attention : t5_h4_d256_k3_do02_lr001_b2048
    tcn            : t25_ch30_lv4_k5_do25_lr01_b2048
    cnn_lstm       : t25_lstm2_f32-16-8_k5_do25_lr01_b2048
    cnn_bilstm     : t25_blstm2_f32-16-8_k5_do25_lr01_b2048
    """
    parts = []

    if timesteps is not None:
        parts.append(f"t{timesteps}")

    if model_type == "conv_attention":
        if num_heads        is not None: parts.append(f"h{num_heads}")
        if d_model          is not None: parts.append(f"d{d_model}")
        if kernel_size      is not None: parts.append(f"k{kernel_size}")

    elif model_type == "tcn":
        if tcn_channel_size is not None: parts.append(f"ch{tcn_channel_size}")
        if tcn_num_levels   is not None: parts.append(f"lv{tcn_num_levels}")
        if kernel_size      is not None: parts.append(f"k{kernel_size}")

    elif model_type == "cnn_lstm":
        if lstm_layers      is not None: parts.append(f"lstm{lstm_layers}")
        if conv_filters     is not None:
            parts.append("f" + "-".join(str(f) for f in conv_filters))
        if kernel_size      is not None: parts.append(f"k{kernel_size}")

    elif model_type == "cnn_bilstm":
        if lstm_layers      is not None: parts.append(f"blstm{lstm_layers}")
        if conv_filters     is not None:
            parts.append("f" + "-".join(str(f) for f in conv_filters))
        if kernel_size      is not None: parts.append(f"k{kernel_size}")

    if dropout    is not None: parts.append(f"do{_do_str(dropout)}")
    if lr         is not None: parts.append(f"lr{_lr_str(lr)}")
    if batch_size is not None: parts.append(f"b{batch_size}")

    return "_".join(parts)


def set_prefix(fold_idx=None, run_name=None, model_type=None,
               use_kfold=False, config_tag=None):
    """
    Build the full run prefix used for ALL file naming.
    Convention: fold1_runname_configtag_modeltype
    Example   : fold1_20260717_131226_t5_h4_d256_lr001_b2048_convattention

    All save/plot functions receive this prefix directly — they never
    call set_prefix again, so there is no duplication.
    """
    parts = []
    if use_kfold and fold_idx is not None:
        parts.append(f"fold{fold_idx + 1}")
    if run_name:
        parts.append(str(run_name))
    if config_tag:
        parts.append(str(config_tag))
    if model_type:
        parts.append(model_type.replace("_", ""))
    return "_".join(parts)


# ------------------------------------------------------------------ #
# Folder helper — still used internally by save functions
# ------------------------------------------------------------------ #

def set_folder_path(use_kfold=False, fold_idx=None,
                    base_dir="results", model_type=None):
    if use_kfold:
        folder_path = os.path.join(base_dir, "kfold", model_type or "")
        if fold_idx is not None:
            folder_path = os.path.join(folder_path, f"fold_{fold_idx + 1}")
    else:
        folder_path = os.path.join(base_dir, "single", model_type or "")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# ------------------------------------------------------------------ #
# Save functions
# All accept 'prefix' directly — no internal set_prefix call.
# ------------------------------------------------------------------ #

def save_model(model, prefix, use_kfold=False, fold_idx=None,
               base_dir="results", model_type=None):
    folder = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    path   = os.path.join(folder, f"{prefix}_model.pt")
    torch.save(model.state_dict(), path)
    return path


def save_results(results_dict, prefix, use_kfold=False, fold_idx=None,
                 base_dir="results", model_type=None):
    folder = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    path   = os.path.join(folder, f"{prefix}_results.pt")
    torch.save(results_dict, path)
    return path


def save_checkpoint(model, optimizer, epoch, prefix, use_kfold=False,
                    fold_idx=None, base_dir="results", model_type=None):
    folder = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    path   = os.path.join(folder, f"{prefix}_ckpt_epoch{epoch}.pt")
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':                epoch,
        'fold_idx':             fold_idx,
    }, path)
    return path


def save_csv(all_metrics, prefix, use_kfold=False, fold_idx=None,
             base_dir="results", model_type=None):
    import pandas as pd
    folder = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    path   = os.path.join(folder, f"{prefix}_metrics.csv")
    pd.DataFrame(all_metrics).to_csv(path, index=False)
    return path


def save_json(epoch_logs, prefix, use_kfold=False, fold_idx=None,
              base_dir="results", model_type=None):
    folder = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    path   = os.path.join(folder, f"{prefix}_epoch_logs.json")
    with open(path, "w") as f:
        json.dump({"prefix": prefix, "epochs": epoch_logs}, f, indent=4)
    return path


# ------------------------------------------------------------------ #
# WandB / experiment logging
# ------------------------------------------------------------------ #

def log_config(wandb_config, prefix, use_kfold, model_type,
               base_dir="results"):
    folder = set_folder_path(use_kfold=use_kfold, model_type=model_type,
                             base_dir=base_dir)
    path = os.path.join(folder, f"{prefix}_config.json")
    with open(path, "w") as f:
        json.dump(wandb_config, f, indent=4)
    if wandb.run is not None:
        artifact = wandb.Artifact(f"{prefix}_config", type="config")
        artifact.add_file(path)
        wandb.run.log_artifact(artifact)


def log_flops(model, prefix, model_type, use_kfold, input_shape,
              base_dir="results", fold_idx=0):
    if fold_idx > 0:
        return
    try:
        import warnings
        from ptflops import get_model_complexity_info
        folder = set_folder_path(use_kfold=use_kfold, model_type=model_type,
                                 base_dir=base_dir)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            macs, params = get_model_complexity_info(
                model, input_res=input_shape,
                as_strings=False, print_per_layer_stat=False, verbose=False
            )
        path = os.path.join(folder, f"{prefix}_flops.json")
        with open(path, "w") as f:
            json.dump({"FLOPs_MACs": macs, "Parameters": params}, f, indent=4)
        if wandb.run is not None:
            artifact = wandb.Artifact(f"{prefix}_flops", type="flops")
            artifact.add_file(path)
            wandb.run.log_artifact(artifact)
            wandb.config.update({"FLOPs_MACs": macs})
        logger.info(f"[FLOPs] {macs} | [Params] {params}")
    except Exception as e:
        logger.warning(f"FLOPs estimation skipped: {e}")
        logger.info(f"[FLOPs] None | [Params] None")


def log_env():
    import sys, platform, sklearn, matplotlib, seaborn
    env_info = {
        "python_version":     sys.version,
        "platform":           platform.platform(),
        "torch_version":      torch.__version__,
        "cuda_available":     torch.cuda.is_available(),
        "cuda_version":       torch.version.cuda if torch.cuda.is_available() else None,
        "sklearn_version":    sklearn.__version__,
        "numpy_version":      np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "seaborn_version":    seaborn.__version__,
        "psutil_version":     psutil.__version__,
    }
    with open("environment_info.json", "w") as f:
        json.dump(env_info, f, indent=2)
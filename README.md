# CAReN: Conv-Attention-Refinement Network for Eye Movement Event Classification

This repository contains the codebase for **CAReN** (Conv-Attention-Refinement Network),
a hybrid 1D-CNN + multi-head self-attention architecture for classifying eye movement
events — fixation, saccade, smooth pursuit (SP), and blink — using a compact 20&nbsp;ms
temporal context window. It also contains faithful reimplementations of several
established baselines (TCN, CNN-LSTM, CNN-BiLSTM, Skip-AttSeqNet) used for comparison,
along with the full ablation study pipeline (temporal context, input features,
architecture components) used to validate the proposed design.

This project builds on and extends the online eye-movement classification codebase of
Elmadjian et al. (2023) — see [Acknowledgements](#acknowledgements--related-work) below.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Supported Models](#supported-models)
- [Training](#training)
- [Full CLI Reference](#full-cli-reference)
- [Ablation Studies](#ablation-studies)
- [Results Directory Structure](#results-directory-structure)
- [Platform Notes](#platform-notes)
- [Reproducibility](#reproducibility)
- [Acknowledgements & Related Work](#acknowledgements--related-work)
- [Citation](#citation)

---

## Repository Structure

```
oemc-attention/
├── main.py                      # Entry point: preprocessing + training + evaluation
├── train.py                     # Core training loop (train_model, main_kfold)
│
├── configs/
│   └── args.py                  # All CLI argument definitions and defaults
│
├── models/
│   ├── __init__.py              # Model registry (get_model, print_summary)
│   ├── conv_attention.py        # CAReN (proposed model)
│   ├── tcn.py                   # TCN baseline (Bai et al. / Elmadjian et al.)
│   ├── cnn_lstm.py               # CNN-LSTM baseline (Startsev et al. / Elmadjian et al.)
│   ├── cnn_bilstm.py             # CNN-BiLSTM baseline (Startsev et al. / Elmadjian et al.)
│   └── skip_attseqnet.py         # Skip-AttSeqNet baseline (Wang et al. 2025, best-effort)
│
├── data/
│   ├── preprocessor.py           # Multi-scale feature extraction (speed, direction,
│   │                              # stddev, displacement)
│   └── dataset.py                # LOADER_REGISTRY (lookahead / lookback windowing)
│
├── utils/
│   ├── helpers.py                # Device selection, checkpointing, config-tag naming
│   ├── metrics.py                # Sample/event-level scoring, CM/ROC/PR plotting
│   └── logger.py                 # Shared logger instance
│
├── ablation/
│   ├── run_ablation.py           # Entry point for all ablation studies
│   ├── feature_ablation.py       # Progressive feature-set ablation
│   ├── timestep_ablation.py      # Temporal context window ablation
│   ├── architecture_ablation.py  # Attention / positional encoding / encoder depth
│   └── build_architecture_tables.py  # Builds paper-ready tables from ablation CSVs
│
├── dataset/
│   ├── data_gazecom/              # Raw GazeCom recordings (see Datasets below)
│   ├── data_hmr/                  # Raw HMR recordings
│   └── processed/                 # Extracted feature tensors (.npz), created by
│                                   # run_preprocessing.py
│
└── results/                       # All training outputs — see Results Directory
                                    # Structure below
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchinfo`, `ptflops`, `wandb`, `scikit-learn`,
`torchmetrics`, `tqdm`, `pandas`, `matplotlib`, `python-dotenv`.

If you plan to log experiments to [Weights & Biases](https://wandb.ai), create a `.env`
file in the project root with:

```
WANDB_API_KEY=your_key_here
```

---

## Datasets

Two datasets are supported out of the box:

| Dataset | Sampling rate | Default preprocessing stride | Notes |
|---|---|---|---|
| **GazeCom** | 250&nbsp;Hz | 10 | Startsev et al. (2019) annotation of Dorr et al.'s recordings |
| **HMR** (Head-Mounted Raw) | 200&nbsp;Hz | 8 | Introduced by Elmadjian et al. (2023) |

Raw data is expected under `dataset/data_gazecom/` and `dataset/data_hmr/` respectively,
following the folder/CSV conventions described in the original OEMC repository (see
[Acknowledgements](#acknowledgements--related-work)).

Adding a new dataset requires implementing a compatible loader in `data/preprocessor.py`
and registering its default `stride`/`frequency` in `configs/args.py::data_defaults()`.

---

## Preprocessing

Before training, raw recordings must be converted into fixed-size feature tensors:

```bash
python run_preprocessing.py --dataset gazecom
python run_preprocessing.py --dataset hmr
```

This extracts four multi-scale features per timestep — **speed**, **direction**,
**standard deviation**, and **displacement** — and writes the result to
`dataset/processed/<dataset>_s<stride>_f<freq>_w<window_length>_o<offset>/`.

`main.py` automatically resolves the correct processed folder based on `--dataset`
(and `--stride`/`--frequency` if overridden), or you can point directly at a folder with
`--data_path` (useful on Kaggle or other environments where the processed data lives
outside the repo, e.g. `--data_path /kaggle/input/.../hmr_s8_f200_w1.0_o0`).

---

## Supported Models

| `--model_type` | Description | Source | LSTM-based? |
|---|---|---|---|
| `conv_attention` | **CAReN** (proposed): Conv embedding → multi-head self-attention → convolutional refinement encoder → classifier | This work | No |
| `tcn` | Temporal Convolutional Network, 4 dilated residual blocks | Bai et al. (2018), as adapted by Elmadjian et al. (2023) | No |
| `cnn_lstm` | 3-layer CNN encoder + 2-layer unidirectional LSTM | Startsev et al. (2019), as adapted by Elmadjian et al. (2023) | Yes |
| `cnn_bilstm` | 3-layer CNN encoder + 2-layer bidirectional LSTM | Startsev et al. (2019), as adapted by Elmadjian et al. (2023) | Yes |
| `skip_attseqnet` | 4-layer CNN encoder + skip connection + 2-layer BiLSTM + attention pooling | Wang et al. (2025), **best-effort reimplementation — no public code available**; see docstring in `models/skip_attseqnet.py` for approximations made | Yes |

### Recommended per-model configuration

| Model | Optimizer | LR | LR schedule | Dropout | Kernel size | Architecture-specific |
|---|---|---|---|---|---|---|
| CAReN | AdamW (wd=1e-4) | 0.001 | CosineAnnealingLR | 0.20 | 3 | `d_model=256`, `num_heads=4` |
| TCN | Adamax | 0.01 | F1-plateau halving\* | 0.25 | 5 | 4 levels × 30 channels |
| CNN-LSTM | RMSprop | 0.01 | F1-plateau halving\* | 0.25 | 5 | LSTM×2, hidden=32 |
| CNN-BiLSTM | RMSprop | 0.01 | F1-plateau halving\* | 0.25 | 5 | BiLSTM×2, hidden=16×2 |
| Skip-AttSeqNet | Adam | 0.001 | CosineAnnealingLR (not reported in paper) | CNN 0.20 / RNN 0.30 | 3 | Filters (32,16,8,8), BiLSTM hidden=16 |

\* *F1-plateau halving* exactly replicates Elmadjian et al.'s manual rule: the learning
rate is halved whenever validation F1-macro plateaus (changes by <0.1 percentage points)
or declines over a 2-epoch lookback (see `--scheduler elmadjian` / `_elmadjian_lr_step` in
`train.py`).

All models share the following training protocol, standardized across baselines for fair
comparison rather than reusing each paper's original budget: **NLLLoss** (no class
weighting), **batch size 2048**, **max 300 epochs** with early stopping (patience 10),
and **Stratified 5-Fold cross-validation** (seed 42) for final reported results.

### Recommended temporal context (`--timesteps`)

Each model is evaluated at its own empirically-supported "native" window, since window
requirements differ by architecture:

| Model | GazeCom (250&nbsp;Hz) | HMR (200&nbsp;Hz) | Basis |
|---|---|---|---|
| CAReN | **5** (20&nbsp;ms) | **4** (20&nbsp;ms) | Section 5.1 ablation — empirical optimum on both datasets |
| TCN / CNN-LSTM / CNN-BiLSTM | **25** (~100&nbsp;ms) | **20** (~100&nbsp;ms) | Elmadjian et al.'s reported ideal window |
| Skip-AttSeqNet | **257** (~1&nbsp;s) | **200** (~1&nbsp;s, analogous duration — not evaluated on HMR in the original paper) | Wang et al., Table 3 "context size" |

> **Note:** Skip-AttSeqNet's 4 VALID-padding convolutional layers (kernel size 3) impose
> a hard minimum of **9 timesteps** — it cannot be evaluated at CAReN's 20&nbsp;ms window.

---

## Training

Basic usage:

```bash
python main.py --dataset <gazecom|hmr> --model_type <model> --timesteps <T> [options...]
```

### Example: train CAReN on GazeCom (20 ms window, 5-fold CV, logged to WandB)

```bash
python main.py --dataset gazecom --model_type conv_attention --timesteps 5 \
    --d_model 256 --num_heads 4 --kernel_size 3 --dropout 0.20 --lr 0.001 \
    --use_kfold --n_splits 5 --use_wandb --checkpoint
```

### Example: train the TCN baseline on HMR (~100 ms window, faithful replication)

```bash
python main.py --dataset hmr --model_type tcn --timesteps 20 \
    --tcn_kernel_size 5 --tcn_channel_size 30 --tcn_num_levels 4 \
    --dropout 0.25 --lr 0.01 --loader_mode lookback \
    --use_kfold --n_splits 5 --use_wandb --checkpoint
```

### Example: train Skip-AttSeqNet on GazeCom (~1 s window)

```bash
python main.py --dataset gazecom --model_type skip_attseqnet --timesteps 257 \
    --kernel_size 3 --lr 0.001 --loader_mode lookback \
    --use_kfold --n_splits 5 --use_wandb --checkpoint
```

### Running only specific folds (e.g. resuming a partial run)

`--start_fold` is 0-indexed; `--max_folds` is how many folds to run starting from there.
To run only fold 3 and fold 4 (out of 5):

```bash
python main.py --dataset gazecom --model_type skip_attseqnet --timesteps 257 \
    --kernel_size 3 --lr 0.001 --loader_mode lookback \
    --use_kfold --n_splits 5 --start_fold 2 --max_folds 2 --use_wandb --checkpoint
```

### Quick sanity check before a full run

```bash
python main.py --dataset gazecom --model_type conv_attention --timesteps 5 -e 5 \
    --use_kfold --n_splits 5
```

(`-e 5` limits training to 5 epochs; omit `--use_wandb`/`--checkpoint` for a fast,
disk-light smoke test.)

---

## Full CLI Reference

| Flag | Default | Description |
|---|---|---|
| `-d, --dataset` | *(required)* | `gazecom` or `hmr` |
| `-m, --model_type` | *(required)* | `conv_attention`, `tcn`, `cnn_lstm`, `cnn_bilstm`, `skip_attseqnet` |
| `--data_path` | auto | Direct path to preprocessed data folder (overrides auto-resolution) |
| `--stride`, `--frequency` | dataset default | Preprocessing overrides |
| `--timesteps` | 5 | Temporal context window length |
| `--d_model` | 256 | Hidden/filter dimension (`conv_attention`) |
| `--num_heads` | 4 | Attention heads (`conv_attention`) |
| `--kernel_size` | 3 | Conv kernel size (non-TCN models) |
| `--dropout` | 0.2 | Dropout (all models except Skip-AttSeqNet, which has its own flags) |
| `--tcn_kernel_size` / `--tcn_channel_size` / `--tcn_num_levels` | 5 / 30 / 4 | TCN-specific |
| `--skip_cnn_dropout` / `--skip_rnn_dropout` | 0.2 / 0.3 | Skip-AttSeqNet-specific dropout |
| `-e, --epochs` | 300 | Max epochs |
| `-b, --batch_size` | 2048 | Mini-batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--patience` | 10 | Early-stopping patience (on val_loss) |
| `--optimizer` | model default | `adamw`, `adamax`, `rmsprop`, `adam` |
| `--scheduler` | model default | `cosine`, `plateau`, `step`, `elmadjian` |
| `--loss` | `nll` | `nll` (plain) or `nll_w` (class-balanced) |
| `--loader_mode` | `lookahead` | `lookahead` or `lookback` (causal) windowing |
| `--use_kfold` | off | Enable Stratified 5-Fold CV (else single 80/20 hold-out) |
| `--n_splits` | 5 | Number of folds |
| `--start_fold` | 0 | First fold index to run (0-indexed) |
| `--max_folds` | = `n_splits` | Number of folds to actually run |
| `--checkpoint` / `--no-checkpoint` | on | Save a checkpoint every epoch |
| `--use_wandb` | off | Log to Weights & Biases |
| `--wandb_project` | auto-named | Override the WandB project name |
| `-r, --run_name` | timestamp | Custom run identifier |

Run `python main.py --help` for the authoritative, up-to-date list.

---

## Ablation Studies

All ablation scripts share a common entry point and always use a single stratified
80/20 hold-out split (not k-fold) for speed across many configurations:

```bash
python ablation/run_ablation.py feature -d gazecom
python ablation/run_ablation.py timestep -d gazecom
python ablation/run_ablation.py architecture -d gazecom --timesteps 5
```

### Feature ablation

Progressive addition of input features, starting from the velocity-based baseline
(speed + direction):

```bash
python ablation/run_ablation.py feature -d gazecom --combos sp_dir sp_dir_std sp_dir_dis sp_dir_std_dis
```

### Timestep ablation

Sweeps the temporal context window at a fixed feature set:

```bash
python ablation/run_ablation.py timestep -d gazecom --values 1 5 10 20 25
python ablation/run_ablation.py timestep -d hmr --values 1 4 8 16 20
```

### Architecture ablation

Toggles self-attention, positional encoding, and convolutional encoder depth for CAReN
(9 variants covering a progressive build-up, leave-one-out, and encoder-depth sweep):

```bash
python ablation/run_ablation.py architecture -d gazecom --timesteps 5
```

### Building paper-ready tables from ablation results

```bash
python ablation/build_architecture_tables.py -d gazecom
```

Generates markdown tables (progressive build-up, leave-one-out, encoder-depth sweep)
directly from the ablation summary CSVs, with no additional training required.

---

## Results Directory Structure

```
results/
├── kfold/
│   └── <dataset>/<model_type>/fold_<N>/
│       ├── <prefix>_model.pt          # state_dict of the best epoch
│       ├── <prefix>_results.pt        # {"preds", "labels", "probs"} of the best epoch
│       ├── <prefix>_metrics.csv       # per-fold summary metrics
│       ├── <prefix>_epoch_logs.json   # full per-epoch history
│       ├── <prefix>_cmcount.png/.pdf  # confusion matrix (sample + event level)
│       ├── <prefix>_cmpercent.png/.pdf
│       ├── <prefix>_roc.png/.pdf
│       └── <prefix>_pr.png/.pdf
├── single/
│   └── <dataset>/<model_type>/         # same file set, single hold-out split
└── ablation/
    └── <model_type>/{features,timesteps,architecture}/<dataset>/<model_type>/
        # per-combination results, plus a top-level
        # <ablation_type>_ablation_<dataset>_summary.csv
```

Both **local files** and a **WandB Artifact** (`model.pt`, `results.pt`,
`epoch_logs.json`, `metrics.csv` bundled together) are produced for every fold, so raw
predictions can be recovered later without re-running inference.

---

## Platform Notes

- **Windows + AMD GPU (torch-directml):** `nn.LSTM` is not supported by DirectML
  (missing `aten::_thnn_fused_lstm_cell`). `cnn_lstm`, `cnn_bilstm`, and
  `skip_attseqnet` are automatically switched to CPU **only when DirectML is the
  detected backend** — CUDA (e.g. Kaggle T4/P100) and MPS are unaffected and will use
  the GPU normally for these models.
- **Windows + large datasets:** `DataLoader(num_workers=...)` defaults to `0` on
  Windows (vs. `4` on Linux) to avoid a `spawn`-based multiprocessing pickling failure
  with large in-memory tensors (`OSError: [Errno 22] Invalid argument`). This does not
  occur on Linux/Kaggle, which uses `fork`.
- **FLOPs/parameter estimation** (`log_flops` / `ptflops`) may fail silently on
  DirectML for the same LSTM reason above; this does not affect training, only the
  logged `[FLOPs]`/`[Params]` values for that run.

---

## Reproducibility

- Global seed **42** is fixed for `torch`, `numpy`, and `random` at the start of every
  run (`set_randomness()`), and used identically for `StratifiedKFold` splitting.
- Stratified (not plain) k-fold is used throughout to preserve class balance across
  folds, given the severe class imbalance in both datasets (SP and Blink are minority
  classes).
- Every run's config is logged to WandB and saved locally as `<prefix>_config.json`.

---

## Acknowledgements & Related Work

This codebase extends the original OEMC (Online Eye-Movement Classification) repository
by Elmadjian et al.:

> Elmadjian, C., Gonzales, C., Costa, R. L. da, & Morimoto, C. H. (2023). Online
> eye-movement classification with temporal convolutional networks. *Behavior Research
> Methods*, 55, 3602–3620. <https://doi.org/10.3758/s13428-022-01978-2>
> Source: <https://github.com/elmadjian/OEMC>

The `cnn_lstm` and `cnn_bilstm` baselines follow:

> Startsev, M., Agtzidis, I., & Dorr, M. (2019). 1D CNN with BLSTM for automated
> classification of fixations, saccades, and smooth pursuits. *Behavior Research
> Methods*, 51, 556–572.

The `tcn` baseline architecture follows:

> Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic
> convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.

The `skip_attseqnet` baseline is a best-effort reimplementation (no public code
available) of:

> Wang, X., Fan, L., Li, H., Bi, X., Jiang, W., & Ma, X. (2025). Skip-AttSeqNet:
> Leveraging skip connection and attention-driven Seq2seq model to enhance eye movement
> event detection in Parkinson's disease. *Biomedical Signal Processing and Control*,
> 99, 106862. <https://doi.org/10.1016/j.bspc.2024.106862>

The GazeCom dataset annotation follows Startsev et al. (2019) above, based on recordings
by Dorr, M., Martinetz, T., Gegenfurtner, K. R., & Barth, E. (2010). The HMR dataset was
introduced by Elmadjian et al. (2023) above.

---

## Citation

If you use CAReN or this codebase in your research, please cite:

```bibtex
@article{caren2026,
  title   = {CAReN: A Convolutional Self-Attention Network with Minimal Temporal
             Context for Eye Movement Event Classification},
  author  = {[Author names]},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026}
}
```

Please also cite the original baseline sources listed above if you use the
corresponding reimplementations in your own comparisons.
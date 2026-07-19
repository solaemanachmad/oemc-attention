# Preprocessing
## How to Run
We have designed run_preprocessing.py to automatically configure the correct default parameters (stride, frequency) based on Elmadjian et al.

1. GazeCom:
python run_preprocessing.py \
    --dataset gazecom \
    --raw_data_path dataset/data_gazecom \
    --processed_data_path dataset/processed

2. HMR:
python run_preprocessing.py \
    --dataset hmr \
    --raw_data_path dataset/data_hmr \
    --processed_data_path dataset/processed

## Customizing Parameters
If you wish to override the default parameters for experimental purposes, you can manually pass the arguments in the terminal:

Bash
python prepare_data.py \
    --dataset gazecom \
    --raw_data_path dataset/data_gazecom \
    --processed_data_path dataset/processed \
    --window_length 1.5 \
    --stride 12 \
    --frequency 250 \
    --offset 0

## Argument Reference

| Argument | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--dataset` | `str` | `gazecom` | Preset choice (`gazecom` or `hmr`). Auto-sets `stride` and `frequency` if not manually defined. Also used to name the output folder. |
| `--raw_data_path` | `str` | `dataset/data_gazecom` | Path to the folder containing raw TSV/CSV dataset files. |
| `--processed_data_path` | `str` | `dataset/processed` | Base directory where the extracted `.npz` folders will be saved. |
| `--window_length` | `float` | `1.0` | Window length (in seconds) for the sliding window logic. |
| `--offset` | `int` | `0` | Offset index used for determining the target label. |
| `--stride` | `int` | *auto* | Sliding window stride size. (Gazecom defaults to 10, HMR to 8). |
| `--frequency` | `int` | *auto* | Sampling frequency in Hz. (Gazecom defaults to 250, HMR to 200). |

## Extracted Features
The preprocessor.py logic applies multiple window sizes (determined by strides) and extracts 4 key spatial-temporal features from the raw X and Y coordinates:

1. Speed (Amplitude / Time): The velocity of the eye movement within the window.

2. Direction (Angle): The trajectory angle calculated using math.atan2.

3. Displacement: The root of the variance of X and Y segments.

4. Standard Deviation (StdDev): The standard deviation of the distance from the mean coordinates.
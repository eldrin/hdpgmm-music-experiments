HDPGMM Music Experiments
========================


## Installation

setting up the environment is to use `conda` and `requirements.txt`. If `conda` or `miniconda`, it can be setup locally (specifically contained in this project directory) by following command.

```{bash}
sh install_env.sh
```

Once `conda` command is available, the python virtual environment can be setup by following command.

```{bash}
conda env create -f environment.yaml
```

## Dataset downloads

## Dataset pre-processing

### Million Song Dataset (the training corpus)


### Downstream datasets
#### GTZAN

```{bash}
python scripts/gtzan_data_prep.py
usage: Data Pre-processing [-h] [--n-fft N_FFT] [--hop-sz HOP_SZ] [--feature {feature,mel}]
                           [--verbose | --no-verbose]
                           gtzan_path out_path
Data Pre-processing: error: the following arguments are required: gtzan_path, out_path
```

`gtzan_path` needs to be set as the top level directory contains the audio files. (i.e., `genres`directory) If there is no specific parameters set, it extracts the "feature" with default setup where `n_fft=2048`, `hop_s=512`, and `mel_len=128`.


#### MagnaTagATune



#### Echonest

HDPGMM Music Experiments
========================


## Installation

setting up the environment is to use `conda` and `environment.yaml`. If `conda` command is available already, the python virtual environment can be setup by following command.

```{bash}
conda env create -f environment.yaml
```

If `conda` or `miniconda` is not available yet, the entire setup process (including setting up the virtual environment) can be done locally by following command.

```{bash}
sh install_env.sh
```

This installation will install separate `miniconda` instance inside this project directory, thus does not technically interfere each other if there is a system-wide `miniconda` preinstalled. But activating the specific conda environment embedded in this project requires custome command.

```{bash}
export project_path=/path/to/project/hdpgmm-music-experiments/
source ${project_path}/miniconda3/bin/activate hdpgmm_music_experiment
```

(this command is executed at the end of the installation script)

### Note on macOS

It seems there is a bug in python in macOS when importing the `soundfile` package. It seems to look up `libsndfile.dylib` on the wrong places (at least on the tested machine of mine). It can be temporarily resolve by export right location of the `libsndfile.dylib`:

```{bash}
brew install libsndfile
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"; python scripts/some_script.py
```

It may not be case that the location is under `/opt/homebrew/lib`. The location can be easily identified and assigned by following command:

```{bash}
libsndfile_loc=$(brew list libsndfile | grep libsndfile.dylib | xargs dirname)
export DYLD_LIBRARY_PATH="${libsndfile_loc}:$DYLD_LIBRARY_PATH"
python scripts/some_script_loads_soundfile.py
```


## Dataset downloads

### Million Song Dataset (the training corpus)
### Downstream datasets
#### GTZAN
#### MagnaTagATune
#### Echonest


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

```{bash}
python scripts/mtat_data_prep.py
usage: MTAT Data Pre-processing [-h] [--n-fft N_FFT] [--hop-sz HOP_SZ] [--verbose | --no-verbose] mtat_path out_path
MTAT Data Pre-processing: error: the following arguments are required: mtat_path, out_path
```

similarly to the `gtzan` script, it assumes a particular file-tree as follows:

```
-- mtat_path/
|-- audio/
| |-- 0/
| |-- 1/
| |.../
| |-- f/
|-- annotations_final.csv
```

(If the data is downloaded by the way above, it automatically organized in such a way.) By default it extracts `features` with same setup as mentioned above.


#### Echonest



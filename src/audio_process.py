from typing import Union
import warnings
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

import librosa
from librosa.core.spectrum import _spectrogram
# from librosa.feature import mfcc, delta, chroma_stft
from librosa.feature import delta, chroma_stft
from librosa.feature import melspectrogram
from librosa.onset import onset_strength
# from librosa import power_to_db

from .utils import knit_path


logger = logging.getLogger(__name__)


def process_track(
    path: Union[str, Path],
    n_fft: int,
    hop_sz: int,
    eps: float=1e-8,
    n_mfcc: int=13
) -> tuple[Path, npt.NDArray, int]:
    """ Process a track to features
    """
    feature_len = 0
    feature = np.array([])
    path = Path(path)  # to make sure
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path)

        # compute (mel) (db) spectrogram (as base feature)
        S, n_fft = _spectrogram(y=y, n_fft=n_fft,
                                hop_length=hop_sz, power=2.)
        s = melspectrogram(S=S, sr=sr)
        s_db = librosa.power_to_db(s)

        # compute the "timbre" feature
        m = librosa.feature.mfcc(S=s_db, sr=sr, n_mfcc=n_mfcc)
        dm = delta(m, order=1)
        ddm = delta(m, order=2)

        # compute harmonic feature
        chrm = chroma_stft(S=S, sr=sr)
        ln_chrm = np.log(np.maximum(chrm, eps))

        # compute rhythm feature
        onset = onset_strength(S=s_db, sr=sr)
        ln_onset = np.log(np.maximum(onset, eps))[None]

        # stitch them
        feature = np.r_[m, dm, ddm, ln_chrm, ln_onset]
        feature_len = feature.shape[1]

    except Exception as e:
        print(e)
        logger.info(f'mp3 file for {path.as_posix()} is corrupted! skipping...')

    return path, feature.T, feature_len


def process_and_save(
    msdid_path: tuple[str, Union[str, Path]],
    out_path: Path,
    n_fft: int,
    hop_sz: int
) -> int:
    """
    """
    # parse input
    msd_id, path = msdid_path
    p = knit_path(msd_id, out_path)

    if p.exists():
        feature = np.load(p, mmap_mode='r')
        feature_len = feature.shape[0]
    else:
        path, feature, feature_len = process_track(path, n_fft, hop_sz)

        # save them if the file is valid
        if feature_len > 0:
            np.save(p, feature)

    return feature_len  # return the length of each melspecs



def process_track_mel(
    path: Path,
    n_fft: int,
    hop_sz: int,
    *args,
    **kwargs
) -> tuple[Path, npt.ArrayLike, int]:
    """ Process a track to features
    """
    feature_len = 0
    feature = np.array([])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path)
        feature = melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_sz)
        feature = librosa.power_to_db(feature)
        feature_len = feature.shape[1]

    except Exception as e:
        logger.info(f'mp3 file for {path.as_posix()} is corrupted! skipping...')

    return path, feature.T, feature_len

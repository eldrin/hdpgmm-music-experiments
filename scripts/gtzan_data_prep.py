from typing import Union
import os
import sys
from pathlib import Path
import argparse
import logging
import warnings
import pickle as pkl
import multiprocessing as mp
from functools import partial

import numpy as np
import numpy.typing as npt

import librosa
import h5py
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())

from src.config import DEFAULTS
from src.audio_process import process_track as _process_track
from src.audio_process import process_track_mel as _process_track_mel


logging.basicConfig()
logger = logging.getLogger("GTZAN_DataPreparation")

MAX_JOBS = os.environ.get('MAX_JOBS')
if MAX_JOBS is None:
    MAX_JOBS = 4
else:
    MAX_JOBS = int(MAX_JOBS)

H5PY_STR = h5py.special_dtype(vlen=str)
FEATURE_EXT = {
    'mel': _process_track_mel,
    'feature': _process_track
}

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Data Pre-processing",
        description=(
            "Processing GTZAN dataset. "
            "This script process the preview songs into "
            "per-song mel spectrogram, and save them into numpy files."
        )
    )

    parser.add_argument("gtzan_path", type=str,
                        help="path where GTZAN dataset is unzipped")
    parser.add_argument("out_path", type=str,
                        help="root directory where the outputs are stored")
    parser.add_argument("--n-fft", type=float, default=DEFAULTS['n_fft'],
                        help="the size of each audio frame, which is to be FFT-ed")
    parser.add_argument("--hop-sz", type=float, default=DEFAULTS['hop_sz'],
                        help="the amount to slide through")
    parser.add_argument("--feature", type=str, default='mfcc13lnchroma',
                        choices=set(FEATURE_EXT.keys()),
                        help="the type of features to be extracted")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def process_track(
    path: Path,
    n_fft: int,
    hop_sz: int,
    feature: str = 'mfcc13lnchroma',
    eps: float=1e-8
) -> tuple[npt.ArrayLike, str]:
    """ Process a track to mel-spectrogram

    It loads the file (in 22kHz, mono channel)
    and process them into mel spectrogram

    we take the logarithm of mel-spectrogram, which makes
    the distribution more suitable for the multi-variate Gaussians
    """
    global FEATURE_EXT
    assert feature in FEATURE_EXT

    # parse genre
    genre = path.name.split('.')[0]
    path, feature, _ = FEATURE_EXT[feature](path, n_fft, hop_sz, eps)
    return path, feature, genre


def main() -> None:
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    gtzan_path = Path(args.gtzan_path)
    cands = list(gtzan_path.glob('*/*.au'))

    # wrap per-file processing function
    out_path = Path(args.out_path) / args.feature
    out_path.mkdir(exist_ok=True, parents=True)
    _process_file = partial(process_track,
                            n_fft = args.n_fft,
                            hop_sz = args.hop_sz,
                            feature = args.feature)

    # compute melspecs and store them
    features = []
    feature_dim = -1
    total_length = 0
    with mp.Pool(processes=MAX_JOBS) as pool:
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:
            for path, feature, genre in pool.imap_unordered(_process_file, cands):
                features.append((path, feature, genre))
                total_length += feature.shape[0]
                if feature_dim == -1:
                    feature_dim = feature.shape[1]
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    dataset_fn = out_path.parent / f'gtzan_{args.feature}.h5'
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        hf.create_dataset('data', (total_length, feature_dim), dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:

            indptr = [0]
            filenames = []
            genres = []
            for i, (path, feature, genre) in enumerate(features):

                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                hf['data'][i0:i1] = feature
                indptr.append(i1)
                filenames.append(path.name)
                genres.append(genre)

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(filenames, dtype=H5PY_STR))
        hf.create_dataset('targets', data=np.array(genres, dtype=H5PY_STR))


if __name__ == "__main__":
    main()

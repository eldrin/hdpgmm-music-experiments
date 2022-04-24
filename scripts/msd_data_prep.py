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
logger = logging.getLogger("MSDDataPreparation")


MSD_MP3_PATH = os.environ.get('MSD_MP3_PATH')
if MSD_MP3_PATH is None:
    raise ValueError('MSD_MP3_PATH is not set!')

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
            "Data preparation for HDP mixture model "
            "powered music representation learning. "
            "The main source of the data is Million Song Dataset, "
            "and this script process the preview songs into "
            "per-song mel spectrogram, and save them into numpy files."
        )
    )

    parser.add_argument("path_info", type=str,
                        help=("pickled dictionary file containing MSD index "
                              "to the subpath of each song"))
    parser.add_argument("out_path", type=str,
                        help="root directory where the outputs are stored")
    parser.add_argument("--n-fft", type=float, default=DEFAULTS['n_fft'],
                        help="the size of each audio frame, which is to be FFT-ed")
    parser.add_argument("--hop-sz", type=float, default=DEFAULTS['hop_sz'],
                        help="the amount to slide through")
    parser.add_argument("--target-list", type=str, default=None,
                        help=("an optional text file contains filenames of target files "
                              "if not given, it samples from entire pool with given "
                              "subsample number"))
    parser.add_argument("--feature", type=str, default='feature',
                        choices=set(FEATURE_EXT.keys()),
                        help="the type of features to be extracted")
    parser.add_argument("--subsample", type=int, default=100000,
                        help="the number of subsample that is going to be processed")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def knit_path(
    msd_id: str,
    out_path: Path
) -> Path:
    """ knit full path using target directory and MSD id
    """
    # build out dir
    out_dir = out_path / msd_id[2] / msd_id[3] / msd_id[4]

    # make folder if not exsists
    out_dir.mkdir(parents=True, exist_ok=True)

    # knit the output filename
    p = out_dir / f'{msd_id}.npy'

    return p


def process_track(
    msdid_path: tuple[str, Union[str, Path]],
    out_path: Path,
    n_fft: int,
    hop_sz: int,
    feature: str='mfcc13lnchroma',
    eps: float=1e-8
) -> int:
    """ Process a track to mel-spectrogram

    It loads the file (in 22kHz, mono channel)
    and process them into mel spectrogram

    we take the logarithm of mel-spectrogram, which makes
    the distribution more suitable for the multi-variate Gaussians

    TODO: MFCC might indeed be a better choice for GMMs,
          although it sacrifices covariance structure,
          which might or might not be a good option for GMMs
          as a representation learner
    """
    global FEATURE_EXT
    assert feature in FEATURE_EXT

    # parse input
    msd_id, path = msdid_path
    p = knit_path(msd_id, out_path)

    if p.exists():
        feature = np.load(p, mmap_mode='r')
        feature_len = feature.shape[0]
    else:
        path, feature, feature_len = FEATURE_EXT[feature](path, n_fft, hop_sz)

        # save them if the file is valid
        if feature_len > 0:
            np.save(p, feature)

    return feature_len  # return the length of each melspecs


def main() -> None:
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    msd2path_fn = Path(args.path_info)
    msd_mp3_path = Path(MSD_MP3_PATH)

    with open(msd2path_fn, 'rb') as fp:
        msd2path = pkl.load(fp)

    # if it's exceeding the entire dataset, can't process
    assert args.subsample <= len(msd2path)

    if args.target_list:
        # we expect MSD ids per line
        with open(args.target_list, 'r') as fp:
            msd_ids = [l.replace('\n','') for l in fp]
        cands = [(i, msd_mp3_path / msd2path[i]) for i in msd_ids]

    else:
        # for sampling, "listyfying" is better
        msd2path = list(msd2path.items())

        # sub sample
        cands = [
            (msd2path[i][0], msd_mp3_path / msd2path[i][1])
            for i in np.random.choice(len(msd2path),
                                      args.subsample,
                                      replace=False)
        ]

    # wrap per-file processing function
    out_path = Path(args.out_path) / args.feature
    out_path.mkdir(exist_ok=True, parents=True)
    _process_file = partial(process_track,
                            out_path = out_path,
                            n_fft = args.n_fft,
                            hop_sz = args.hop_sz,
                            feature = args.feature)

    # compute melspecs and store them
    total_length = 0
    with mp.Pool(processes=MAX_JOBS) as pool:
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:
            for l in pool.imap_unordered(_process_file, cands):
                total_length += l
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    fns = list(out_path.glob('*/*/*/*.npy'))
    dataset_fn = out_path.parent / f'msd_{args.feature}.h5'
    feature_dim = np.load(fns[0]).shape[1]
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        hf.create_dataset('data', (total_length, feature_dim), dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:

            indptr = [0]
            msd_ids = []
            for i, (msd_id, path) in enumerate(cands):
                p = knit_path(msd_id, out_path)
                if not p.exists():
                    continue

                feature = np.load(p)

                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                hf['data'][i0:i1] = feature
                indptr.append(i1)
                msd_ids.append(msd_id)

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(msd_ids, dtype=H5PY_STR))


if __name__ == "__main__":
    main()

from typing import Union
import os
import sys
from pathlib import Path
import argparse
import logging
import pickle as pkl
import multiprocessing as mp
import warnings
from functools import partial

import numpy as np

import h5py
from tqdm import tqdm

import librosa
from librosa.core.spectrum import _spectrogram
from librosa.feature import mfcc, delta, chroma_stft
from librosa.feature import melspectrogram
from librosa.onset import onset_strength

import torch
from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)

sys.path.append(Path(__file__).parent.parent.as_posix())

from src.config import DEFAULTS

logging.basicConfig()
logger = logging.getLogger("MSDDataPreparation")

# controlling torch multithreading
torch.set_num_threads(1)

MSD_MP3_PATH = os.environ.get('MSD_MP3_PATH')
if MSD_MP3_PATH is None:
    raise ValueError('MSD_MP3_PATH is not set!')

MAX_JOBS = os.environ.get('MAX_JOBS')
if MAX_JOBS is None:
    MAX_JOBS = 4
else:
    MAX_JOBS = int(MAX_JOBS)

H5PY_STR = h5py.special_dtype(vlen=str)

np.random.seed(2022)

# FROM CLMR paper
AUG_COEF_DEFAULT = {
    'transforms_polarity': 0.8,
    'transforms_noise': 0.01,
    'transforms_gain': 0.3,
    'transforms_filters': 0.8,
    'transforms_delay': 0.3,
    'transforms_pitch': 0.6,
    'transforms_reverb': 0.6
}
SAMPLING_RATE = 22050

TRANSFORM = ComposeMany(
    [
        RandomApply([PolarityInversion()],
                    p=AUG_COEF_DEFAULT['transforms_polarity']),
        RandomApply([Noise()], p=AUG_COEF_DEFAULT['transforms_noise']),
        RandomApply([Gain()], p=AUG_COEF_DEFAULT['transforms_gain']),
        RandomApply(
            [HighLowPass(sample_rate=SAMPLING_RATE)],
            p=AUG_COEF_DEFAULT['transforms_filters']
        ),
        RandomApply([Delay(sample_rate=SAMPLING_RATE)],
                    p=AUG_COEF_DEFAULT['transforms_delay']),
        # # it's tricky as we run variable length clip
        # RandomApply(
        #     [
        #         PitchShift(
        #             n_samples=SAMPLING_RATE * 30,
        #             sample_rate=SAMPLING_RATE,
        #         )
        #     ],
        #     p=AUG_COEF_DEFAULT['transforms_pitch'],
        # ),
        RandomApply(
            [Reverb(sample_rate=SAMPLING_RATE)],
            p=AUG_COEF_DEFAULT['transforms_reverb']
        ),
    ],
    num_augmented_samples=1
)


def parse_arguments() -> argparse.Namespace:
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
    parser.add_argument("--subsample", type=int, default=100000,
                        help="the number of subsample that is going to be processed")
    parser.add_argument("--n-augs", type=int, default=10,
                        help="the number of data augmentation")
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


def __process_track(
    msdid_path: tuple[str, Union[str, Path]],
    out_path: Path,
    n_fft: int,
    hop_sz: int,
    n_mfcc: int=13,
    apply_augmentation: bool = False,
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
    # parse input
    msd_id, path = msdid_path
    path = Path(path)
    p = knit_path(msd_id, out_path)

    if p.exists():
        return np.load(p).shape[1]

    feature_len = 0
    feature = np.array([])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path)

        # AUGMENTATION
        if apply_augmentation:
            with torch.no_grad():
                y = torch.from_numpy(y)[None]
                y = TRANSFORM(y).detach().cpu().numpy()[0, 0]

        # compute (mel) (db) spectrogram (as base feature)
        S, n_fft = _spectrogram(y=y, n_fft=n_fft,
                                hop_length=hop_sz, power=2.)
        s = melspectrogram(S=S, sr=sr)
        s_db = librosa.power_to_db(s)

        # compute the "timbre" feature
        m = mfcc(S=s_db, sr=sr, n_mfcc=n_mfcc)
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

    # save them if the file is valid
    if feature_len > 0:
        np.save(p, feature)

    return feature_len  # return the length of each melspecs


def process_track(
    msdid_path: tuple[str, Union[str, Path]],
    out_path: Path,
    n_fft: int,
    hop_sz: int,
    n_mfcc: int=13,
    n_augs: int=10,
    eps: float=1e-8
) -> int:
    """
    """
    # process the original
    feature_len = __process_track(
        msdid_path, out_path, n_fft, hop_sz, n_mfcc, False, eps
    )

    # process the augmented files
    for i in range(n_augs):
        out_path_aug = out_path.parent / f'{out_path.name}_aug{i:d}'
        __process_track(
            msdid_path, out_path_aug, n_fft, hop_sz, n_mfcc, True, eps
        )

    return feature_len


def write_data(
    p: Union[str, Path],
    dataset: h5py.Dataset,
    indptr: list[int]
) -> int:
    """
    """
    feature = np.load(p).T

    # get index pointers
    i0 = indptr[-1]
    i1 = i0 + feature.shape[0]

    # write data
    dataset[i0:i1] = feature

    return i1 - i0  # feature length


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
    out_path = Path(args.out_path) / 'feature_org'
    out_path.mkdir(exist_ok=True, parents=True)
    _process_file = partial(process_track,
                            out_path = out_path,
                            n_fft = args.n_fft,
                            hop_sz = args.hop_sz,
                            n_augs = args.n_augs)

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

    # TODO: read all the files including the augs?
    #       might be much easier to re-write the whole thing
    fns = list(out_path.glob('*/*/*/*.npy'))
    dataset_fn = out_path.parent / f'msd200k_feature_aug.h5'
    feature_dim = np.load(fns[0]).T.shape[1]
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        dataset_shape = (total_length, feature_dim)
        hf.create_dataset('data', dataset_shape, dtype='f4')
        data_aug = hf.create_group('augmentation')

        # make datasets for the augmentations
        for i in range(args.n_augs):
            data_aug.create_dataset(f'{i:d}', dataset_shape, dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:

            indptr = [0]
            msd_ids = []
            for msd_id, _ in cands:
                p = knit_path(msd_id, out_path)
                if not p.exists():
                    continue

                feature_len = write_data(p, hf['data'], indptr)
                for i in range(args.n_augs):
                    out_path_aug = out_path.parent / f'{out_path.name}_aug{i:d}'
                    p = knit_path(msd_id, out_path_aug)
                    feature_len_ = write_data(p, data_aug[f'{i:d}'], indptr)
                    assert feature_len == feature_len_

                indptr.append(indptr[-1] + feature_len)
                msd_ids.append(msd_id)

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(msd_ids, dtype=H5PY_STR))


if __name__ == "__main__":
    main()

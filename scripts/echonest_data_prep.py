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
from src.experiment.echonest import _load_interaction
from msd_data_prep import process_track, knit_path, FEATURE_EXT


logging.basicConfig()
logger = logging.getLogger("LastFM1k_DataPreparation")


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


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echonest_data_prep",
        description=(
            "Processing Echonest/MSD dataset. "
            "This script process the preview songs into "
            "per-song feature, and save them into numpy files."
        )
    )

    parser.add_argument("msd_path_info", type=str,
                        help="file contains the map from MSD id to the subpath of the audio (.pkl)")
    parser.add_argument("msd_song2track", type=str,
                        help="file contains the map from MSD song id to the track id (.pkl)")
    parser.add_argument("echonest_triplet", type=str,
                        help="filename of Echonest/MSD interaction triplet data (.txt)")
    parser.add_argument("out_path", type=str,
                        help="root directory where the outputs are stored")
    parser.add_argument("--n-fft", type=float, default=DEFAULTS['n_fft'],
                        help="the size of each audio frame, which is to be FFT-ed")
    parser.add_argument("--hop-sz", type=float, default=DEFAULTS['hop_sz'],
                        help="the amount to slide through")
    parser.add_argument("--feature", type=str, default='feature',
                        choices=set(FEATURE_EXT.keys()),
                        help="the type of features to be extracted")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def main() -> None:
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    # load msdid to subpath
    with Path(args.msd_path_info).open('rb') as fp:
        msd2path = pkl.load(fp)

    # build the interaction matrix
    # build target 
    mat, userlist, itemlist = _load_interaction(
        args.echonest_triplet,
        args.msd_song2track
    )
    msd_song_path = Path(MSD_MP3_PATH)

    cands = [
        (i, track, msd_song_path / msd2path[track])
        for i, track in enumerate(itemlist)
        if track in msd2path and (msd_song_path / msd2path[track]).exists()
    ]
    cands_cols, itemlist, cands = zip(*cands)
    cands = list(zip(itemlist, cands))

    # if there's missing track, remove that column
    # in the interaction matrix
    mat = mat[:, cands_cols].tocsr()

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
    dataset_fn = out_path.parent / f'echonest_{args.feature}.h5'
    feature_dim = np.load(fns[0]).shape[1]
    with h5py.File(dataset_fn, 'w') as hf:

        # save them into the dataset
        interaction = hf.create_group('interaction')
        interaction.create_dataset('indptr', data=mat.indptr)
        interaction.create_dataset('indices', data=mat.indices)
        interaction.create_dataset('data', data=mat.data)
        interaction.create_dataset('userlist',
                                   data=np.array(userlist, dtype=H5PY_STR))
        interaction.create_dataset('itemlist',
                                   data=np.array(itemlist, dtype=H5PY_STR))

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

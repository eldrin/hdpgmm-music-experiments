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
import pandas as pd

from scipy import sparse as sp

import librosa
import h5py
from tqdm import tqdm

sys.path.append(Path(__file__).parent.parent.as_posix())

from src.config import DEFAULTS
from src.audio_process import process_track


logging.basicConfig()
logger = logging.getLogger("MTATDataPreparation")

MAX_JOBS = os.environ.get('MAX_JOBS')
if MAX_JOBS is None:
    MAX_JOBS = 4
else:
    MAX_JOBS = int(MAX_JOBS)

H5PY_STR = h5py.special_dtype(vlen=str)

np.random.seed(2022)


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="MTAT Data Pre-processing",
        description=(
            "Processing MTAT dataset. "
            "This script process the preview songs into "
            "per-song mel spectrogram, and save them into numpy files."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("mtat_path", type=str,
                        help="""path where MagnaTagATune dataset is unzipped.
we assume the folder contains both music audio files
and the annotation file (annotations_final.csv).
the structure the script is expecting is as follows:\n
  -- mtat_path/
   |-- audio/
   | |-- 0/
   | |-- 1/
   | |.../
   | |-- f/
   |-- annotations_final.csv

""")
    parser.add_argument("out_path", type=str,
                        help="root directory where the outputs are stored")
    parser.add_argument("--n-fft", type=float, default=DEFAULTS['n_fft'],
                        help="the size of each audio frame, which is to be FFT-ed")
    parser.add_argument("--hop-sz", type=float, default=DEFAULTS['hop_sz'],
                        help="the amount to slide through")
    parser.add_argument('--verbose', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="set verbosity")
    return parser.parse_args()


def parse_annotations(
    annot: pd.DataFrame
) -> tuple[sp.csr_matrix, list[str]]:
    """
    """
    idx2tag = annot.columns[1:-1].tolist()
    tag2idx = {t:i for i, t in enumerate(idx2tag)}

    indptr = [0]
    indices = []
    for i in range(annot.shape[0]):
        tags = np.where(annot.iloc[i, 1:-1])[0].tolist()
        indptr.append(indptr[-1] + len(tags))
        indices.extend(tags)
    data = [1] * len(indices)

    X = sp.csr_matrix((data, indices, indptr),
                      shape=(len(indptr) - 1, len(idx2tag)))
    return X, idx2tag


def main() -> None:
    """
    """
    args = parse_arguments()
    if args.verbose:
        logger.setLevel(logging.INFO)

    ann = pd.read_csv(Path(args.mtat_path) / "annotations_final.csv", sep='\t')
    tag_mat, idx2tag = parse_annotations(ann)
    cands = [Path(args.mtat_path) / "audio" / fn for fn in ann['mp3_path'].tolist()]
    fn2id = {Path(fn).name: i for fn, i in ann[['mp3_path', 'clip_id']].values}

    # wrap per-file processing function
    out_path = Path(args.out_path)
    _process_file = partial(process_track,
                            n_fft = args.n_fft,
                            hop_sz = args.hop_sz)

    # compute melspecs and store them
    features = []
    feature_dim = -1
    total_length = 0
    with mp.Pool(processes=MAX_JOBS) as pool:
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:
            for path, feature, feature_len in pool.imap_unordered(_process_file, cands):
                features.append((path, feature, feature_len))
                total_length += feature.shape[0]
                if feature_dim == -1:
                    feature_dim = feature.shape[1]
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    dataset_fn = out_path / 'mtat_feature.h5'
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        hf.create_dataset('data', (total_length, feature_dim), dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not args.verbose) as prog:

            indptr = [0]
            filenames = []
            ids = []
            for i, (path, feature, feature_len) in enumerate(features):
                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                hf['data'][i0:i1] = feature
                indptr.append(i1)
                filenames.append(path.name)
                ids.append(fn2id[filenames[-1]])

                prog.update()

        # write data
        # 1. write audio / ids
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(ids), dtype='i4')
        hf.create_dataset('filenames', data=np.array(filenames, dtype=H5PY_STR))

        # 2. write the tagging
        # as indices are "locally" shuffled by the multi-processing (imap_unordered)
        # we fix the order based on the new id order
        cur_id2row = {idx: j for j, idx in enumerate(ann['clip_id'].values)}
        permutation = [cur_id2row[i] for i in ids]
        tag_mat = tag_mat[permutation].tocsr()  # we shuffle the sparse matrix

        hf.create_group('annotations')
        hf['annotations'].create_dataset('indptr', data=tag_mat.indptr, dtype='i4')
        hf['annotations'].create_dataset('indices', data=tag_mat.indices, dtype='i4')
        hf['annotations'].create_dataset('data', data=tag_mat.data, dtype='f4')
        hf['annotations'].create_dataset('tags', data=np.array(idx2tag, dtype=H5PY_STR))


if __name__ == "__main__":
    main()

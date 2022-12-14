from typing import Optional
from pathlib import Path
import logging
import pickle as pkl
import multiprocessing as mp
from functools import partial

import numpy as np

import h5py
from tqdm import tqdm

from ..audio_process import process_and_save
from ..utils import knit_path
from . import H5PY_STR


logging.basicConfig()
logger = logging.getLogger(__name__)


def process(
    msd_mp3_path: str,
    msd_path_info: str,
    out_path: str,
    out_name: str = 'echonest_feature',
    target_list: Optional[str] = None,
    n_fft: int = 2048,
    hop_sz: int = 512,
    subsample: int = 200_000,
    n_jobs: int = 1,
    verbose: bool = False
) -> None:
    """
    """
    if verbose:
        logger.setLevel(logging.INFO)

    msd2path_fn = Path(msd_path_info)
    msd_mp3_path_ = Path(msd_mp3_path)

    with msd2path_fn.open('rb') as fp:
        msd2path = pkl.load(fp)

    # if it's exceeding the entire dataset, can't process
    assert subsample <= len(msd2path)

    if target_list is not None:
        # we expect MSD ids per line
        with Path(target_list).open('r') as fp:
            msd_ids = [l.replace('\n','') for l in fp]
        cands = [(i, msd_mp3_path_ / msd2path[i]) for i in msd_ids]

    else:
        # for sampling, "listyfying" is better
        msd2path = list(msd2path.items())

        # sub sample
        cands = [
            (msd2path[i][0], msd_mp3_path_ / msd2path[i][1])
            for i in np.random.choice(len(msd2path),
                                      subsample,
                                      replace=False)
        ]

    # wrap per-file processing function
    out_path_ = Path(out_path)
    out_path_.mkdir(exist_ok=True, parents=True)
    _process_file = partial(process_and_save,
                            out_path = out_path_,
                            n_fft = n_fft,
                            hop_sz = hop_sz)

    # compute melspecs and store them
    total_length = 0
    with mp.Pool(processes=n_jobs) as pool:
        with tqdm(total=len(cands), ncols=80,
                  disable=not verbose) as prog:
            for l in pool.imap_unordered(_process_file, cands):
                total_length += l
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    fns = list(out_path_.glob('*/*/*/*.npy'))
    dataset_fn = out_path_.parent / f'{out_name}.h5'
    feature_dim = np.load(fns[0]).shape[1]
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        dataset = hf.create_dataset('data', (total_length, feature_dim),
                                    dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80, disable=not verbose) as prog:

            indptr = [0]
            msd_ids = []
            for msd_id, _ in cands:
                p = knit_path(msd_id, out_path_)
                if not p.exists():
                    continue

                feature = np.load(p)

                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                dataset[i0:i1] = feature
                indptr.append(i1)
                msd_ids.append(msd_id)

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(msd_ids, dtype=H5PY_STR))

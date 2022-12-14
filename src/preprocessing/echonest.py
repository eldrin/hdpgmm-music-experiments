from pathlib import Path
import logging
import pickle as pkl
import multiprocessing as mp
from functools import partial

import numpy as np

import h5py
from tqdm import tqdm


from ..experiment.echonest import _load_interaction
from ..audio_process import process_and_save
from ..utils import knit_path
from . import H5PY_STR


logging.basicConfig()
logger = logging.getLogger(__name__)


def process(
    msd_mp3_path: str,
    msd_path_info: str,
    msd_song2track: str,
    echonest_triplet: str,
    out_path: str,
    out_name: str = 'echonest_feature',
    n_fft: int = 2048,
    hop_sz: int = 512,
    n_jobs: int = 1,
    verbose: bool = False
) -> None:
    """
    """
    out_path_ = Path(out_path)

    if verbose:
        logger.setLevel(logging.INFO)

    # load msdid to subpath
    with Path(msd_path_info).open('rb') as fp:
        msd2path = pkl.load(fp)

    # build the interaction matrix
    # build target
    mat, userlist, itemlist = _load_interaction(
        echonest_triplet,
        msd_song2track
    )
    msd_song_path = Path(msd_mp3_path)

    cands = [
        (track, msd_song_path / msd2path[track])
        for track in itemlist
        if track in msd2path and (msd_song_path / msd2path[track]).exists()
    ]
    cands_cols, itemlist, cands = zip(*cands)
    cands = list(zip(itemlist, cands))

    # if there's missing track, remove that column
    # in the interaction matrix
    mat = mat[:, cands_cols].tocsr()

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
        dataset = hf.create_dataset('data', (total_length, feature_dim),
                                    dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not verbose) as prog:

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

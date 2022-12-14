from pathlib import Path
import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import numpy.typing as npt

import h5py
from tqdm import tqdm

from ..audio_process import process_track as _process_track
from . import H5PY_STR


logging.basicConfig()
logger = logging.getLogger(__name__)


def process_track(
    path: Path,
    n_fft: int,
    hop_sz: int,
    eps: float=1e-8
) -> tuple[Path, npt.NDArray, str]:
    """ Process a track to mel-spectrogram

    It loads the file (in 22kHz, mono channel)
    and process them into mel spectrogram

    we take the logarithm of mel-spectrogram, which makes
    the distribution more suitable for the multi-variate Gaussians
    """
    # parse genre
    genre = path.name.split('.')[0]
    path, feature, _ = _process_track(path, n_fft, hop_sz, eps)
    return path, feature, genre


def process(
    gtzan_path: str,
    out_path: str,
    out_name: str = 'gtzan_feature',
    n_fft: int = 2048,
    hop_sz: int = 512,
    n_jobs: int = 1,
    verbose: bool = False
) -> None:
    """
    """
    if verbose:
        logger.setLevel(logging.INFO)

    gtzan_path_ = Path(gtzan_path)
    cands = list(gtzan_path_.glob('*/*.au'))

    # wrap per-file processing function
    out_path_ = Path(out_path)
    out_path_.mkdir(exist_ok=True, parents=True)
    _process_file = partial(process_track,
                            n_fft = n_fft,
                            hop_sz = hop_sz)

    # compute melspecs and store them
    features = []
    feature_dim = -1
    total_length = 0
    with mp.Pool(processes=n_jobs) as pool:
        with tqdm(total=len(cands), ncols=80,
                  disable=not verbose) as prog:
            for path, feature, genre in pool.imap_unordered(_process_file, cands):
                features.append((path, feature, genre))
                total_length += feature.shape[0]
                if feature_dim == -1:
                    feature_dim = feature.shape[1]
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    dataset_fn = out_path_.parent / f'{out_name}.h5'
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        dataset = hf.create_dataset('data',
                                    (total_length, feature_dim),
                                    dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80, disable=not verbose) as prog:

            indptr = [0]
            filenames = []
            genres = []
            for path, feature, genre in features:

                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                dataset[i0:i1] = feature
                indptr.append(i1)
                filenames.append(path.name)
                genres.append(genre)

                prog.update()

        # write data
        hf.create_dataset('indptr', data=np.array(indptr), dtype='i4')
        hf.create_dataset('ids', data=np.array(filenames, dtype=H5PY_STR))
        hf.create_dataset('targets', data=np.array(genres, dtype=H5PY_STR))

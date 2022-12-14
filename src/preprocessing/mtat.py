from pathlib import Path
import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from scipy import sparse as sp

import h5py
from tqdm import tqdm

from ..audio_process import process_track


logging.basicConfig()
logger = logging.getLogger(__name__)

H5PY_STR = h5py.special_dtype(vlen=str)
URLS = {
    'file1': 'https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001',
    'file2': 'https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002',
    'file3': 'https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003'
}


def parse_annotations(
    annot: pd.DataFrame
) -> tuple[sp.csr_matrix, list[str]]:
    """
    """
    idx2tag = annot.columns[1:-1].tolist()
    # tag2idx = {t:i for i, t in enumerate(idx2tag)}

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


def process(
    mtat_path: str,
    out_path: str,
    out_name: str,
    n_fft: int = 2048,
    hop_sz: int = 512,
    n_jobs: int = 1,
    verbose: bool = False
) -> None:
    """
    """
    if args.verbose:
        logger.setLevel(logging.INFO)

    ann = pd.read_csv(Path(mtat_path) / "annotations_final.csv", sep='\t')
    tag_mat, idx2tag = parse_annotations(ann)
    cands = [Path(mtat_path) / "audio" / fn for fn in ann['mp3_path'].tolist()]
    fn2id = {Path(fn).name: i for fn, i in ann[['mp3_path', 'clip_id']].values}

    # wrap per-file processing function
    out_path_ = Path(out_path)
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
            for path, feature, feature_len in pool.imap_unordered(_process_file,
                                                                  cands):
                features.append((path, feature, feature_len))
                total_length += feature.shape[0]
                if feature_dim == -1:
                    feature_dim = feature.shape[1]
                prog.update()

    # organize final dataset (in hdf)
    # build CSR-like multi-dimensional dataset
    dataset_fn = out_path_ / f'{out_name}.h5'
    with h5py.File(dataset_fn, 'w') as hf:

        # make a dataset
        dataset = hf.create_dataset('data',
                                    (total_length, feature_dim),
                                    dtype='f4')

        # now fill them with pre-computed data
        with tqdm(total=len(cands), ncols=80,
                  disable=not verbose) as prog:

            indptr = [0]
            filenames = []
            ids = []
            for path, feature, feature_len in features:
                # get index pointers
                i0 = indptr[-1]
                i1 = i0 + feature.shape[0]

                # write data
                dataset[i0:i1] = feature
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

        annot = hf.create_group('annotations')
        annot.create_dataset('indptr', data=tag_mat.indptr, dtype='i4')
        annot.create_dataset('indices', data=tag_mat.indices, dtype='i4')
        annot.create_dataset('data', data=tag_mat.data, dtype='f4')
        annot.create_dataset('tags', data=np.array(idx2tag, dtype=H5PY_STR))
